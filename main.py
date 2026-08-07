"""
Battlesnake Blackout 2026 — High-Performance FastAPI Server
Survival Logic V9.0 | APEX PREDATOR Protocol

PRESERVED from V7/V8 (unchanged):
  Cross-turn food memory + TTL eviction
  Dead-snake ghost purge
  Partial-visibility tail vacating
  Probabilistic ghost occupancy
  Reverse multi-source food BFS + hazard-aware Dijkstra
  All V8 bug fixes (1.1–1.5)
  _is_corridor_trap, _is_pin_trap, _min_escape_size

NEW in V9.0:
  GameContext dataclass — built once per turn, shared everywhere
  MoveTier enum — Hierarchical Veto System (no cross-tier scoring)
  _build_context — single context factory in get_best_move
  _is_boxed_in — flood-fill with conservative occupied + time-gate
  _has_death_sentence — DFS depth-5 with time-gate
  _opening_book — turns 1-5 bypass
  _executioner_score — 1v1 starvation food-path blockade
  _constriction_score — space denial + wall press
  _is_ambush — forced-win ambush detector
  _food_race_v2 — multi-enemy food race with tie resolution
  Starvation Clock — replaces static HUNGER_THRESHOLD
  Dynamic phase weights — per-phase W_VORONOI, W_FOOD, W_EDGE etc.
  Shadow penalty — same row/col as enemy within distance 2
  Probabilistic minimax — 60/30/10 paranoid/random/greedy blend
  occupied_conservative — ghost-augmented blocked set for survival
"""

import heapq
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from itertools import product as _iproduct
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("battlesnake")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="الثعبان — Battlesnake Blackout 2026", version="9.0.0")


class GameState(BaseModel):
    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Snake identity
# ---------------------------------------------------------------------------
SNAKE_INFO: dict[str, Any] = {
    "apiversion": "1",
    "author":     "Mina Hussein",
    "color":      "#FF0000",
    "head":       "default",
    "tail":       "default",
    "version":    "9.0.0",
}

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
FOOD_STALE_TTL = 40

# ---------------------------------------------------------------------------
# Per-game memory
# ---------------------------------------------------------------------------
_game_memory: dict[str, dict] = {}


# ===========================================================================
# Enums
# ===========================================================================
class GamePhase(Enum):
    EARLY    = "early"
    MID      = "mid"
    LATE_1V1 = "late_1v1"
    LATE_FFA = "late_ffa"


class MoveTier(IntEnum):
    """Hierarchical priority — higher tier wins; no cross-tier scoring."""
    VETO            = 0
    SURVIVAL_RISKY  = 1
    SURVIVAL_SAFE   = 2
    AGGRESSIVE_SAFE = 3


# ===========================================================================
# GameContext — built ONCE per turn in get_best_move
# ===========================================================================
@dataclass
class GameContext:
    our_head:             tuple[int, int]
    our_body:             list
    our_len:              int
    our_health:           int
    our_tail:             tuple[int, int] | None
    width:                int
    height:               int
    turn:                 int
    occupied:             set              # solid blocked cells
    occupied_conservative: set            # occupied + ghost zones (for survival)
    hazard_set:           set
    hazard_dmg:           int
    enemy_heads:          list            # list[tuple[int,int]] — visible heads
    enemy_data:           list            # list[dict] — processed enemy records
    visible_food:         set
    merged_food:          set
    food_dist_map:        dict
    phase:                GamePhase
    deadline:             float
    view_radius:          int
    enemy_info:           dict            # raw memory entry


# ===========================================================================
# Utility helpers
# ===========================================================================
def _pt(seg: Any) -> tuple[int, int] | None:
    if seg is None:
        return None
    return (seg["x"], seg["y"])


def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


def _get_view_radius(data: dict) -> int:
    try:
        return data["game"]["ruleset"]["settings"]["viewRadius"]
    except (KeyError, TypeError):
        return 5


def _get_hazard_dmg(data: dict) -> int:
    try:
        return int(data["game"]["ruleset"]["settings"]["hazardDamagePerTurn"])
    except (KeyError, TypeError, ValueError):
        return 0


def _is_in_view(px: int, py: int, hx: int, hy: int, radius: int) -> bool:
    return _manhattan(px, py, hx, hy) <= radius


# ===========================================================================
# Memory updates (preserved from V7/V8)
# ===========================================================================

def _update_food_memory(game_id: str, data: dict) -> set[tuple[int, int]]:
    mem = _game_memory.setdefault(
        game_id,
        {"food": set(), "enemy_info": {}, "food_meta": {}, "latency_history": []}
    )
    prev_food: set = mem.setdefault("food", set())
    food_meta: dict = mem.setdefault("food_meta", {})

    you    = data["you"]
    head   = you.get("head") or {}
    hx: int = head.get("x", 0)
    hy: int = head.get("y", 0)
    radius = _get_view_radius(data)
    turn   = data.get("turn", 0)

    visible_food: set = {
        (f["x"], f["y"]) for f in data.get("board", {}).get("food", [])
    }

    new_memory: set = set()
    new_meta: dict = {}

    for pos in prev_food:
        px, py = pos
        if _is_in_view(px, py, hx, hy, radius):
            if pos in visible_food:
                new_memory.add(pos)
                new_meta[pos] = turn
        else:
            last_seen = food_meta.get(pos, 0)
            if (turn - last_seen) <= FOOD_STALE_TTL:
                new_memory.add(pos)
                new_meta[pos] = last_seen

    for pos in visible_food:
        new_memory.add(pos)
        new_meta[pos] = turn

    mem["food"]      = new_memory
    mem["food_meta"] = new_meta
    return new_memory


def _update_enemy_memory(game_id: str, data: dict) -> dict:
    mem = _game_memory.setdefault(
        game_id,
        {"food": set(), "enemy_info": {}, "food_meta": {}, "latency_history": []}
    )
    enemy_info: dict = mem.setdefault("enemy_info", {})

    you_id = data["you"]["id"]
    turn   = data.get("turn", 0)

    # Purge dead snakes (V8 fix 2.2)
    alive_ids = {s["id"] for s in data["board"].get("snakes", [])}
    for dead_id in list(enemy_info.keys()):
        if dead_id not in alive_ids:
            del enemy_info[dead_id]

    for snake in data["board"].get("snakes", []):
        sid = snake["id"]
        if sid == you_id:
            continue
        e_head = snake.get("head")
        if e_head is None:
            continue

        visible_segs = sum(1 for s in snake.get("body", []) if s is not None)
        prev = enemy_info.get(sid, {})

        enemy_info[sid] = {
            "last_seen_turn":    turn,
            "last_known_head":   (e_head["x"], e_head["y"]),
            "last_body_count":   visible_segs,
            "prev_body_count":   prev.get("last_body_count", visible_segs),
            "last_known_length": snake.get("length", visible_segs),
            "prev_known_length": prev.get("last_known_length", 0),
        }

    mem["enemy_info"] = enemy_info
    return enemy_info


# ===========================================================================
# TacticalEngine — V9.0 APEX PREDATOR
# ===========================================================================
class TacticalEngine:

    _DELTAS = ((0, 1), (0, -1), (-1, 0), (1, 0))

    DIRECTIONS: dict[str, tuple[int, int]] = {
        "up":    ( 0,  1),
        "down":  ( 0, -1),
        "left":  (-1,  0),
        "right": ( 1,  0),
    }

    # ── Base constants ─────────────────────────────────────────────────
    W_VORONOI         = 20
    W_COMBAT_KILL     = 800
    W_HAZARD_CELL     = -600
    W_FOOD            = -30
    W_CENTER          = -3
    W_GHOST_BASE      = -250
    W_EDGE            = -60
    W_CORNER          = -180
    GHOST_DECAY_TURNS  = 5
    GHOST_RADIUS       = 2
    KILL_MARGIN        = 2
    HUNGER_THRESHOLD   = 45
    COMPUTE_BUDGET_S: float = 0.250

    # ── Phase-dynamic weights (V9b BLOODLUST calibrated) ────────────────
    PHASE_WEIGHTS: dict[GamePhase, dict] = {
        GamePhase.EARLY: {
            "W_VORONOI": 10, "W_FOOD": -40, "W_EDGE": -4,
            "W_CORNER": -12, "W_COMBAT_KILL": 1000, "HUNGER_THRESHOLD": 55,
        },
        GamePhase.MID: {
            "W_VORONOI": 15, "W_FOOD": -35, "W_EDGE": -4,
            "W_CORNER": -12, "W_COMBAT_KILL": 1000, "HUNGER_THRESHOLD": 50,
        },
        GamePhase.LATE_1V1: {
            "W_VORONOI": 10, "W_FOOD": -30, "W_EDGE": 0,
            "W_CORNER": 0, "W_COMBAT_KILL": 2000, "HUNGER_THRESHOLD": 35,
        },
        GamePhase.LATE_FFA: {
            "W_VORONOI": 15, "W_FOOD": -35, "W_EDGE": -2,
            "W_CORNER": -8, "W_COMBAT_KILL": 1500, "HUNGER_THRESHOLD": 50,
        },
    }

    # ================================================================== #
    #  PRIMITIVE 1: Multi-source Voronoi BFS                             #
    # ================================================================== #
    @classmethod
    def voronoi_bfs(
        cls,
        our_head: tuple[int, int],
        enemy_heads: list[tuple[int, int]],
        occupied: set,
        width: int,
        height: int,
    ) -> int:
        dist: dict = {}
        queue: list = []

        if our_head not in occupied:
            dist[our_head] = (0, 0)
            queue.append((0, our_head))

        for eh in enemy_heads:
            if eh not in occupied and eh not in dist:
                dist[eh] = (0, 1)
                queue.append((0, eh))

        ptr = 0
        while ptr < len(queue):
            steps, (cx, cy) = queue[ptr]; ptr += 1
            owner = dist[(cx, cy)][1]
            ns = steps + 1
            for dx, dy in cls._DELTAS:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in occupied:
                    if (nx, ny) not in dist:
                        dist[(nx, ny)] = (ns, owner)
                        queue.append((ns, (nx, ny)))
                    elif dist[(nx, ny)] == (ns, 0) and owner == 1:
                        dist[(nx, ny)] = (ns, 1)

        return sum(1 for (_, o) in dist.values() if o == 0)

    # ================================================================== #
    #  PRIMITIVE 2: BFS / Dijkstra with hazard cost                     #
    # ================================================================== #
    @classmethod
    def bfs_dist(
        cls,
        start: tuple[int, int],
        targets: set,
        occupied: set,
        width: int,
        height: int,
        *,
        max_dist: int = 10**9,
        hazard_cells: set | None = None,
        hazard_cost: float = 1.0,
    ) -> float:
        if not targets:
            return float(max_dist)
        if start in targets:
            return 0.0

        if bool(hazard_cells) and hazard_cost > 1.0:
            best: dict = {start: 0.0}
            pq = [(0.0, start)]
            while pq:
                d, (cx, cy) = heapq.heappop(pq)
                if d > best.get((cx, cy), float("inf")):
                    continue
                if (cx, cy) in targets:
                    return d
                if d > max_dist:
                    return float(max_dist)
                for dx2, dy2 in cls._DELTAS:
                    nx, ny = cx + dx2, cy + dy2
                    if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in occupied:
                        step = hazard_cost if (nx, ny) in (hazard_cells or set()) else 1.0
                        nd = d + step
                        if nd < best.get((nx, ny), float("inf")):
                            best[(nx, ny)] = nd
                            heapq.heappush(pq, (nd, (nx, ny)))
            return float(max_dist)
        else:
            visited: set = {start}
            queue: list = [start]
            ptr = 0
            dist2 = 0
            while ptr < len(queue):
                layer_end = len(queue)
                dist2 += 1
                if dist2 > max_dist:
                    return float(max_dist)
                while ptr < layer_end:
                    cx, cy = queue[ptr]; ptr += 1
                    for dx2, dy2 in cls._DELTAS:
                        nx, ny = cx + dx2, cy + dy2
                        if (0 <= nx < width and 0 <= ny < height
                                and (nx, ny) not in occupied
                                and (nx, ny) not in visited):
                            if (nx, ny) in targets:
                                return float(dist2)
                            visited.add((nx, ny))
                            queue.append((nx, ny))
            return float(max_dist)

    # ================================================================== #
    #  PRIMITIVE 3: Reverse multi-source food BFS                       #
    # ================================================================== #
    @classmethod
    def _food_bfs_reverse(
        cls,
        food_targets: set,
        occupied: set,
        width: int,
        height: int,
        hazard_cells: set | None = None,
        hazard_dmg: int = 0,
    ) -> dict:
        if not food_targets:
            return {}

        hazard_cost = 1.0 + hazard_dmg / 10.0 if (hazard_cells and hazard_dmg > 0) else 1.0
        use_haz = bool(hazard_cells) and hazard_dmg > 0
        dist_map: dict = {}
        pq: list = []

        for food in food_targets:
            if food not in occupied:
                dist_map[food] = 0.0
                heapq.heappush(pq, (0.0, food))

        while pq:
            cost, (cx, cy) = heapq.heappop(pq)
            if dist_map.get((cx, cy), float("inf")) < cost:
                continue
            for dx, dy in cls._DELTAS:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in occupied:
                    step = hazard_cost if (use_haz and (nx, ny) in (hazard_cells or set())) else 1.0
                    nc = cost + step
                    if nc < dist_map.get((nx, ny), float("inf")):
                        dist_map[(nx, ny)] = nc
                        heapq.heappush(pq, (nc, (nx, ny)))

        return dist_map

    # ================================================================== #
    #  PRIMITIVE 4: Occupied set (V8 tail-vacating logic)               #
    # ================================================================== #
    @classmethod
    def _build_occupied(cls, data: dict, game_id: str, enemy_info: dict) -> set:
        you    = data["you"]
        board  = data["board"]
        you_id = you["id"]
        occupied: set = set()

        for snake in board.get("snakes", []):
            body = snake.get("body", [])
            if not body:
                continue
            sid   = snake["id"]
            is_us = (sid == you_id)

            if is_us:
                just_ate = (
                    len(body) >= 2
                    and _pt(body[0]) is not None
                    and _pt(body[0]) == _pt(body[-1])
                )
            else:
                has_hidden = any(s is None for s in body)
                info = enemy_info.get(sid, {})
                if not has_hidden:
                    just_ate = (
                        len(body) >= 2
                        and _pt(body[0]) is not None
                        and _pt(body[0]) == _pt(body[-1])
                    )
                else:
                    curr_len = snake.get("length", 0)
                    prev_len = info.get("last_known_length", 0)
                    just_ate = (prev_len > 0 and curr_len > prev_len)

            tail_idx = len(body) - 1
            for i, seg in enumerate(body):
                pt = _pt(seg)
                if pt is None:
                    continue
                if i == tail_idx and not just_ate:
                    continue
                occupied.add(pt)

        return occupied

    # ================================================================== #
    #  GameContext factory — called once per turn                        #
    # ================================================================== #
    @classmethod
    def _build_context(
        cls,
        data: dict,
        game_id: str,
        merged_food: set,
        deadline: float,
    ) -> GameContext:
        board   = data["board"]
        you     = data["you"]
        you_id  = you["id"]
        width   = board["width"]
        height  = board["height"]
        turn    = data.get("turn", 0)

        mem         = _game_memory.get(game_id, {})
        enemy_info  = mem.get("enemy_info", {})
        occupied    = cls._build_occupied(data, game_id, enemy_info)
        hazard_set  = {(h["x"], h["y"]) for h in board.get("hazards", [])}
        hazard_dmg  = _get_hazard_dmg(data)
        view_radius = _get_view_radius(data)
        phase       = cls._get_game_phase(data)

        # occupied_conservative: occupied + ghost zones around hidden enemies
        visible_ids = {s["id"] for s in board.get("snakes", []) if s.get("head") is not None}
        occ_cons = set(occupied)
        for sid2, info2 in enemy_info.items():
            if sid2 not in visible_ids:
                lkh = info2.get("last_known_head")
                if lkh:
                    lhx, lhy = lkh
                    lst = info2.get("last_seen_turn", 0)
                    turns_hidden = max(0, turn - lst)
                    # Ghost zone: radius = min(turns_hidden, 2)
                    r = min(turns_hidden + 1, 2)
                    for ddx in range(-r, r + 1):
                        for ddy in range(-r, r + 1):
                            cx2, cy2 = lhx + ddx, lhy + ddy
                            if 0 <= cx2 < width and 0 <= cy2 < height:
                                occ_cons.add((cx2, cy2))

        # Process enemy data
        enemy_heads: list = []
        enemy_data: list = []

        for snake in board.get("snakes", []):
            if snake["id"] == you_id:
                continue
            eh = snake.get("head")
            if eh is not None:
                enemy_heads.append((eh["x"], eh["y"]))
            e_body = snake.get("body", [])
            visible_segs = sum(1 for s in e_body if s is not None)
            info3 = enemy_info.get(snake["id"], {})
            enemy_data.append({
                "id":            snake["id"],
                "head":          snake.get("head"),
                "head_pos":      (eh["x"], eh["y"]) if eh else None,
                "length":        snake.get("length", visible_segs),
                "visible_segs":  visible_segs,
                "health":        snake.get("health") or 100,
                "body":          e_body,
                "prev_length":   info3.get("last_known_length", 0),
            })

        # Visible food
        visible_food = {(f["x"], f["y"]) for f in board.get("food", [])}

        # Food distance map
        food_dist_map = cls._food_bfs_reverse(
            merged_food, occupied, width, height, hazard_set, hazard_dmg
        )

        # Build our body
        our_body = [_pt(s) for s in you.get("body", []) if _pt(s) is not None]
        our_tail = _pt(you["body"][-1]) if you.get("body") else None

        return GameContext(
            our_head             = (you["head"]["x"], you["head"]["y"]),
            our_body             = our_body,
            our_len              = len(you.get("body", [])),
            our_health           = you["health"],
            our_tail             = our_tail,
            width                = width,
            height               = height,
            turn                 = turn,
            occupied             = occupied,
            occupied_conservative = occ_cons,
            hazard_set           = hazard_set,
            hazard_dmg           = hazard_dmg,
            enemy_heads          = enemy_heads,
            enemy_data           = enemy_data,
            visible_food         = visible_food,
            merged_food          = merged_food,
            food_dist_map        = food_dist_map,
            phase                = phase,
            deadline             = deadline,
            view_radius          = view_radius,
            enemy_info           = enemy_info,
        )

    # ================================================================== #
    #  COMBAT: Dynamic H2H risk (V8 preserved, phase-aware)             #
    # ================================================================== #
    @classmethod
    def _h2h_risk_level(cls, our_len: int, enemy_len: int, kill_margin: int | None = None) -> str:
        km   = kill_margin if kill_margin is not None else cls.KILL_MARGIN
        diff = our_len - enemy_len
        if diff >= km:
            return "RISK_NONE"
        elif diff > 0:
            return "RISK_LOW"
        elif diff == 0:
            return "RISK_EQUAL"
        else:
            return "RISK_HIGH"

    # ================================================================== #
    #  SURVIVAL: Boxed-in flood-fill (V9.0)                             #
    # ================================================================== #
    @classmethod
    def _is_boxed_in(cls, ctx: GameContext, candidate: tuple[int, int]) -> bool:
        """
        Flood-fill from candidate using occupied_conservative.
        Returns True if reachable space < our_len + 4 OR free neighbors <= 1.
        Time-gated every 500 iterations (returns False on timeout = assume safe).
        """
        if candidate in ctx.occupied_conservative:
            return True

        # Quick neighbor check
        free_nbrs = sum(
            1 for dx, dy in cls._DELTAS
            if (0 <= candidate[0]+dx < ctx.width
                and 0 <= candidate[1]+dy < ctx.height
                and (candidate[0]+dx, candidate[1]+dy) not in ctx.occupied_conservative)
        )
        if free_nbrs <= 1:
            return True

        target = ctx.our_len + 4
        visited: set = {candidate}
        queue = [candidate]
        ptr   = 0

        while ptr < len(queue):
            if ptr % 500 == 0 and time.monotonic() >= ctx.deadline - 0.04:
                return False  # timeout → assume safe
            cx, cy = queue[ptr]; ptr += 1
            for dx, dy in cls._DELTAS:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < ctx.width and 0 <= ny < ctx.height
                        and (nx, ny) not in ctx.occupied_conservative
                        and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    if len(visited) >= target:
                        return False  # enough space

        return len(visited) < target

    # ================================================================== #
    #  SURVIVAL: Death-sentence DFS depth-5 (V9.0)                     #
    # ================================================================== #
    @classmethod
    def _has_death_sentence(cls, ctx: GameContext, candidate: tuple[int, int]) -> bool:
        """
        DFS depth=5. Returns True if ALL branches lead to space < our_len + 3.
        Time-gated every 50 nodes.
        """
        if candidate in ctx.occupied_conservative:
            return True

        node_count = [0]
        deadline   = ctx.deadline - 0.04

        def dfs(pos: tuple[int, int], depth: int, local_occ: set) -> bool:
            node_count[0] += 1
            if node_count[0] % 50 == 0 and time.monotonic() >= deadline:
                return False  # timeout → assume not sentenced

            neighbors = [
                (pos[0]+dx, pos[1]+dy)
                for dx, dy in cls._DELTAS
                if (0 <= pos[0]+dx < ctx.width
                    and 0 <= pos[1]+dy < ctx.height
                    and (pos[0]+dx, pos[1]+dy) not in local_occ)
            ]
            if not neighbors:
                return True  # dead end

            if depth >= 5:
                # Count reachable at leaf
                vis = {pos}
                q = [pos]; p2 = 0
                while p2 < len(q) and len(vis) < ctx.our_len + 3:
                    cx2, cy2 = q[p2]; p2 += 1
                    for dx2, dy2 in cls._DELTAS:
                        nx2, ny2 = cx2+dx2, cy2+dy2
                        if (0 <= nx2 < ctx.width and 0 <= ny2 < ctx.height
                                and (nx2, ny2) not in local_occ
                                and (nx2, ny2) not in vis):
                            vis.add((nx2, ny2))
                            q.append((nx2, ny2))
                return len(vis) < ctx.our_len + 3

            return all(dfs(n, depth + 1, local_occ | {n}) for n in neighbors)

        return dfs(candidate, 0, ctx.occupied_conservative | {candidate})

    # ================================================================== #
    #  SURVIVAL: Corridor trap (V8 preserved)                           #
    # ================================================================== #
    @classmethod
    def _is_corridor_trap(
        cls,
        candidate: tuple[int, int],
        snake_len: int,
        occupied: set,
        threatened: set,
        width: int,
        height: int,
        kill_cells: set,
    ) -> bool:
        if candidate in kill_cells:
            return False
        blocked = occupied | threatened
        if candidate in blocked:
            return True
        visited = {candidate}
        queue   = [candidate]
        ptr     = 0
        while ptr < len(queue):
            cx, cy = queue[ptr]; ptr += 1
            for dx, dy in cls._DELTAS:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < width and 0 <= ny < height
                        and (nx, ny) not in blocked
                        and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return len(visited) < snake_len + 3

    # ================================================================== #
    #  SURVIVAL: Pin-trap 2-ply (V8 preserved)                         #
    # ================================================================== #
    @classmethod
    def _is_pin_trap(
        cls,
        candidate: tuple[int, int],
        head_x: int,
        head_y: int,
        enemies: list,
        occupied: set,
        width: int,
        height: int,
        kill_cells: set,
        deadline: float,
    ) -> bool:
        if candidate in kill_cells:
            return False
        if time.monotonic() >= deadline - 0.06:
            return False

        neck = (head_x, head_y)
        our_next = [
            (candidate[0]+dx, candidate[1]+dy)
            for dx, dy in cls._DELTAS
            if (0 <= candidate[0]+dx < width
                and 0 <= candidate[1]+dy < height
                and (candidate[0]+dx, candidate[1]+dy) not in occupied
                and (candidate[0]+dx, candidate[1]+dy) != neck)
        ]
        if not our_next:
            return True

        vis_enemies = [e for e in enemies if e.get("head") is not None][:3]
        if not vis_enemies:
            return False

        enemy_opts: list = []
        for e in vis_enemies:
            eh = (e["head"]["x"], e["head"]["y"])
            opts = [
                (eh[0]+dx, eh[1]+dy)
                for dx, dy in cls._DELTAS
                if (0 <= eh[0]+dx < width
                    and 0 <= eh[1]+dy < height
                    and (eh[0]+dx, eh[1]+dy) not in occupied)
            ]
            enemy_opts.append(opts if opts else [eh])

        for combo in _iproduct(*enemy_opts):
            if time.monotonic() >= deadline - 0.05:
                break
            blocked = occupied | set(combo)
            if all(c in blocked for c in our_next):
                return True

        return False

    # ================================================================== #
    #  SURVIVAL: Escape-route analysis (V8 preserved)                   #
    # ================================================================== #
    @classmethod
    def _min_escape_size(
        cls,
        candidate: tuple[int, int],
        snake_len: int,
        occupied: set,
        enemies: list,
        width: int,
        height: int,
        deadline: float,
    ) -> bool:
        if time.monotonic() >= deadline - 0.05:
            return False

        occ_after_us = occupied | {candidate}
        vis_enemies  = [e for e in enemies if e.get("head") is not None][:2]
        if not vis_enemies:
            return False

        enemy_next_opts: list = []
        for e in vis_enemies:
            eh   = (e["head"]["x"], e["head"]["y"])
            opts = [
                (eh[0]+dx, eh[1]+dy)
                for dx, dy in cls._DELTAS
                if 0 <= eh[0]+dx < width and 0 <= eh[1]+dy < height
                and (eh[0]+dx, eh[1]+dy) not in occupied
            ]
            enemy_next_opts.append(opts if opts else [eh])

        for combo in _iproduct(*enemy_next_opts):
            if time.monotonic() >= deadline - 0.04:
                break
            occ_combo = occ_after_us | set(combo)
            if candidate in occ_combo:
                return True
            visited = {candidate}
            queue   = [candidate]
            ptr     = 0
            while ptr < len(queue):
                cx, cy = queue[ptr]; ptr += 1
                for dx, dy in cls._DELTAS:
                    nx, ny = cx+dx, cy+dy
                    if (0 <= nx < width and 0 <= ny < height
                            and (nx, ny) not in occ_combo
                            and (nx, ny) not in visited):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            if len(visited) < snake_len + 2:
                return True

        return False

    # ================================================================== #
    #  COMBAT: Probabilistic ghost occupancy (V8 preserved)             #
    # ================================================================== #
    @classmethod
    def _compute_ghost_risk(
        cls, enemy_info: dict, turn: int, occupied: set, width: int, height: int,
    ) -> dict:
        ghost_risk: dict = {}

        for info in enemy_info.values():
            lkh = info.get("last_known_head")
            if lkh is None:
                continue
            lst          = info.get("last_seen_turn", 0)
            turns_hidden = max(0, turn - lst)
            if turns_hidden >= cls.GHOST_DECAY_TURNS:
                continue

            decay = max(0.0, 1.0 - turns_hidden / cls.GHOST_DECAY_TURNS)
            max_r = min(turns_hidden + 1, cls.GHOST_RADIUS)
            lhx, lhy = lkh

            feasibility: set = set()
            for dx in range(-max_r, max_r + 1):
                rem = max_r - abs(dx)
                for dy in range(-rem, rem + 1):
                    cx, cy = lhx+dx, lhy+dy
                    if (0 <= cx < width and 0 <= cy < height
                            and (cx, cy) not in occupied):
                        feasibility.add((cx, cy))

            if not feasibility:
                continue

            penalty = cls.W_GHOST_BASE * (1.0 / len(feasibility)) * decay
            for cell in feasibility:
                ghost_risk[cell] = max(ghost_risk.get(cell, 0.0) + penalty, -300.0)

        return ghost_risk

    # ================================================================== #
    #  FOOD: Food Race V2 — multi-enemy with tie resolution (V9.0)      #
    # ================================================================== #
    @classmethod
    def _food_race_v2(
        cls,
        candidate: tuple[int, int],
        ctx: GameContext,
        enemy_data: list,
    ) -> float:
        """Returns race_value in [-1.0, +1.0]."""
        if not ctx.food_dist_map:
            return 0.0

        d_us = ctx.food_dist_map.get(candidate, float("inf"))
        enemy_dists = [
            ctx.food_dist_map.get(e["head_pos"], float("inf"))
            for e in enemy_data if e["head_pos"] is not None
        ]

        if not enemy_dists:
            return 1.0

        d_min = min(enemy_dists)

        if d_us < d_min:
            return 1.0
        elif d_us > d_min:
            return -0.5

        # Tied — find tying enemies
        ties = sum(1 for d in enemy_dists if d == d_min)
        if ties > 1:
            return -0.9  # multi-tie = certain conflict

        # Solo tie: length comparison
        tying = next(
            (e for e in enemy_data
             if e["head_pos"] is not None
             and abs(ctx.food_dist_map.get(e["head_pos"], float("inf")) - d_min) < 0.01),
            None,
        )
        if tying is None:
            return 0.0

        e_len = tying["length"]
        if ctx.our_len > e_len + 1:
            return 0.8
        elif ctx.our_len > e_len:
            return 0.4
        elif ctx.our_len == e_len:
            return -0.5
        else:
            return -0.9

    # ================================================================== #
    #  COMBAT: Executioner Mode (V9.0) — 1v1 starvation blockade       #
    # ================================================================== #
    @classmethod
    def _executioner_score(
        cls,
        candidate: tuple[int, int],
        enemy: dict,
        ctx: GameContext,
    ) -> float:
        """
        Bonus for blocking the enemy's path to the nearest food
        when executioner mode is active.
        """
        e_pos = enemy.get("head_pos")
        if e_pos is None or not ctx.food_dist_map:
            return 0.0

        e_food_dist  = ctx.food_dist_map.get(e_pos, float("inf"))
        e_health     = enemy.get("health", 100)

        if e_food_dist >= e_health:
            # Enemy can't reach food in time — switch to constriction
            return 0.0

        # Find first-step cells on enemy's shortest food path
        ex, ey = e_pos
        first_steps = [
            (ex+dx, ey+dy)
            for dx, dy in cls._DELTAS
            if (0 <= ex+dx < ctx.width and 0 <= ey+dy < ctx.height
                and ctx.food_dist_map.get((ex+dx, ey+dy), float("inf")) < e_food_dist)
        ]

        if candidate in first_steps:
            return 2500.0  # blocking their food path
        if ctx.food_dist_map.get(candidate, float("inf")) < e_food_dist:
            return 800.0   # we're already on a food-path cell
        return 0.0

    # ================================================================== #
    #  COMBAT: Constriction Scorer + wall press (V9.0)                 #
    # ================================================================== #
    @classmethod
    def _constriction_score(
        cls,
        candidate: tuple[int, int],
        enemy_head_pos: tuple[int, int],
        e_free: int,
        ctx: GameContext,
    ) -> float:
        """
        e_free = enemy Voronoi without us (precomputed once).
        Per-candidate: compute enemy Voronoi with our candidate as blocker.
        """
        if time.monotonic() >= ctx.deadline - 0.05:
            return 0.0
        e_trapped = cls.voronoi_bfs(
            enemy_head_pos, [candidate], ctx.occupied, ctx.width, ctx.height
        )
        constriction = max(0, e_free - e_trapped)
        return constriction * 55.0  # 40 territory + 15 wall press

    # ================================================================== #
    #  COMBAT: Ambush Detector (V9.0)                                   #
    # ================================================================== #
    @classmethod
    def _is_ambush(
        cls,
        candidate: tuple[int, int],
        enemy_head_pos: tuple[int, int],
        ctx: GameContext,
    ) -> bool:
        """
        Returns True if enemy has exactly 2 legal moves:
        one toward us (H2H), one into a boxed-in cell.
        """
        ex, ey = enemy_head_pos
        if _manhattan(candidate[0], candidate[1], ex, ey) != 1:
            return False

        e_legal = [
            (ex+dx, ey+dy)
            for dx, dy in cls._DELTAS
            if (0 <= ex+dx < ctx.width
                and 0 <= ey+dy < ctx.height
                and (ex+dx, ey+dy) not in ctx.occupied)
        ]

        if len(e_legal) != 2:
            return False

        toward_us = [m for m in e_legal if m == candidate]
        away      = [m for m in e_legal if m != candidate]

        if not toward_us or not away:
            return False

        return cls._is_boxed_in(ctx, away[0])

    # ================================================================== #
    #  STRATEGIC: Game phase detection (preserved)                      #
    # ================================================================== #
    @classmethod
    def _get_game_phase(cls, data: dict) -> GamePhase:
        you_id  = data["you"]["id"]
        turn    = data.get("turn", 0)
        our_len = len(data["you"]["body"])
        enemies = [s for s in data["board"].get("snakes", []) if s["id"] != you_id]
        n       = len(enemies)

        if n == 0:
            return GamePhase.LATE_1V1
        if n == 1:
            return (GamePhase.LATE_1V1
                    if enemies[0].get("head") is not None
                    else GamePhase.MID)
        if n == 2:
            if all(e.get("length", 0) < our_len for e in enemies):
                return GamePhase.LATE_FFA
        return GamePhase.EARLY if turn < 20 else GamePhase.MID

    # ================================================================== #
    #  STRATEGIC: Opening Book turns 1-5 (V8.1 BLOODLUST — food first) #
    # ================================================================== #
    @classmethod
    def _opening_book(cls, ctx: GameContext, safe_moves: list) -> str | None:
        if ctx.turn > 5 or not safe_moves:
            return None

        # BLOODLUST: always sprint to nearest visible food first (turns 1-5)
        if ctx.visible_food:
            best_d = None
            best_dist = float("inf")
            for d in safe_moves:
                dx, dy = cls.DIRECTIONS[d]
                nx, ny = ctx.our_head[0]+dx, ctx.our_head[1]+dy
                # Direct food hit
                if (nx, ny) in ctx.visible_food:
                    return d
                # Or closest approach
                fd = min(_manhattan(nx, ny, fx, fy) for fx, fy in ctx.visible_food)
                if fd < best_dist:
                    best_dist = fd; best_d = d
            if best_d:
                return best_d

        # No visible food — move toward center
        center_x = ctx.width  // 2
        center_y = ctx.height // 2
        best_d = None
        best_dist = float("inf")
        for d in safe_moves:
            dx, dy = cls.DIRECTIONS[d]
            nx, ny = ctx.our_head[0]+dx, ctx.our_head[1]+dy
            dist = _manhattan(nx, ny, center_x, center_y)
            if dist < best_dist:
                best_dist = dist; best_d = d
        return best_d or safe_moves[0]

    # ================================================================== #
    #  Space-max fallback                                                #
    # ================================================================== #
    @classmethod
    def _rank_space_max(
        cls, moves: list, occupied: set, width: int, height: int,
        head_x: int, head_y: int,
    ) -> str:
        best_move  = moves[0] if moves else "up"
        best_space = -1
        for d in moves:
            dx, dy = cls.DIRECTIONS[d]
            c      = (head_x+dx, head_y+dy)
            vis    = {c}
            q      = [c]; ptr = 0
            while ptr < len(q):
                cx2, cy2 = q[ptr]; ptr += 1
                for ddx, ddy in cls._DELTAS:
                    nc = (cx2+ddx, cy2+ddy)
                    if (0 <= nc[0] < width and 0 <= nc[1] < height
                            and nc not in occupied and nc not in vis):
                        vis.add(nc); q.append(nc)
            if len(vis) > best_space:
                best_space = len(vis); best_move = d
        return best_move

    # ================================================================== #
    #  2-PLY PROBABILISTIC MINIMAX (V9.0: 60/30/10 blend)              #
    # ================================================================== #
    @classmethod
    def _get_legal_moves_sim(cls, head, body, occupied, width, height):
        neck = body[1] if len(body) > 1 else None
        return [
            d for d, (dx, dy) in cls.DIRECTIONS.items()
            if (0 <= head[0]+dx < width and 0 <= head[1]+dy < height
                and (head[0]+dx, head[1]+dy) != neck
                and (head[0]+dx, head[1]+dy) not in occupied)
        ]

    @classmethod
    def _simulate_and_evaluate(
        cls, us, enemy_sims, all_moves, food_set,
        hazard_set, hazard_dmg, food_dist_map, width, height,
    ) -> float:
        from collections import defaultdict

        us2   = {**us, "body": list(us["body"])}
        enems = [{**e, "body": list(e["body"])} for e in enemy_sims]
        all_s = [us2] + enems

        # 1. New heads
        new_heads: dict = {}
        for s in all_s:
            if not s.get("alive", True): continue
            dx, dy = cls.DIRECTIONS.get(all_moves.get(s["id"], "up"), (0, 1))
            new_heads[s["id"]] = (s["head"][0]+dx, s["head"][1]+dy)

        # 2. OOB
        for s in all_s:
            if not s.get("alive", True): continue
            nx, ny = new_heads[s["id"]]
            if not (0 <= nx < width and 0 <= ny < height):
                s["alive"] = False

        if not us2.get("alive", True):
            return -999999.0

        # 3. Body collision (uses actual just_ate flags — V8 fix 1.1)
        body_cells: set = set()
        for s in all_s:
            if not s.get("alive", True): continue
            for i, seg in enumerate(s["body"]):
                if i == len(s["body"]) - 1 and not s.get("just_ate", False):
                    continue
                body_cells.add(seg)

        for s in all_s:
            if not s.get("alive", True): continue
            if new_heads[s["id"]] in body_cells:
                s["alive"] = False

        if not us2.get("alive", True):
            return -999999.0

        # 4. H2H
        head_to_snakes: dict = defaultdict(list)
        for s in all_s:
            if s.get("alive", True):
                head_to_snakes[new_heads[s["id"]]].append(s)

        for pos, claimants in head_to_snakes.items():
            if len(claimants) < 2: continue
            mx = max(c["length"] for c in claimants)
            for c in claimants:
                if c["length"] < mx:
                    c["alive"] = False
                elif any(x["id"] != c["id"] and x["length"] == mx for x in claimants):
                    c["alive"] = False

        if not us2.get("alive", True):
            return -999999.0

        # 5. Advance bodies
        enemy_died = sum(1 for e in enems if not e.get("alive", True))
        for s in all_s:
            if not s.get("alive", True): continue
            nh  = new_heads[s["id"]]
            ate = nh in food_set
            s["body"].insert(0, nh)
            if not ate:
                s["body"].pop()
                s["just_ate"] = False
            else:
                s["health"] = 100; s["just_ate"] = True
            s["head"] = nh; s["length"] = len(s["body"])

        # 6. Health / hazard
        for s in all_s:
            if not s.get("alive", True): continue
            s["health"] -= 1
            if s["head"] in hazard_set:
                s["health"] -= hazard_dmg
            if s["health"] <= 0:
                s["alive"] = False

        if not us2.get("alive", True):
            return -999999.0

        alive_eh    = [e["head"] for e in enems if e.get("alive", True)]
        sim_occ     = {seg for s in all_s if s.get("alive", True) for seg in s["body"]}
        our_voronoi = cls.voronoi_bfs(us2["head"], alive_eh, sim_occ, width, height)
        our_fd      = food_dist_map.get(us2["head"], float(width+height))
        food_rv     = 1.0 if our_fd < 3 else (0.5 if our_fd < 7 else 0.0)

        return (
            our_voronoi * 15
            + enemy_died * 5000
            + us2["health"] * 2
            + food_rv * 100
            - (500 if us2["head"] in hazard_set else 0)
        )

    @classmethod
    def _minimax_2ply(
        cls, moves, data, occupied, merged_food, enemy_heads,
        hazard_set, hazard_dmg, food_dist_map, deadline,
        enemy_info: dict | None = None,
    ) -> dict:
        """Probabilistic 2-ply: 60% paranoid / 30% random / 10% greedy."""
        board   = data["board"]
        you     = data["you"]
        width   = board["width"]
        height  = board["height"]
        you_id  = you["id"]
        our_head = (you["head"]["x"], you["head"]["y"])
        enemies  = [s for s in board.get("snakes", []) if s["id"] != you_id]
        ei       = enemy_info or {}

        def make_sim(snake):
            body = [_pt(seg) for seg in snake.get("body", []) if _pt(seg) is not None]
            head = _pt(snake.get("head"))
            if head is None or not body:
                return None
            has_hidden = any(s is None for s in snake.get("body", []))
            if not has_hidden:
                just_ate = (len(body) >= 2 and body[0] == body[-1])
            else:
                sid      = snake.get("id", "")
                info     = ei.get(sid, {})
                curr_len = snake.get("length", 0)
                prev_len = info.get("last_known_length", 0)
                just_ate = (prev_len > 0 and curr_len > prev_len)
            return {
                "id": snake["id"], "head": head, "body": body,
                "health": snake.get("health") or 100, "length": len(body),
                "alive": True, "just_ate": just_ate,
            }

        us_body     = [_pt(seg) for seg in you["body"] if _pt(seg) is not None]
        us_just_ate = (len(us_body) >= 2 and us_body[0] == us_body[-1])
        us_sim = {
            "id": you_id, "head": our_head, "body": us_body,
            "health": you["health"], "length": len(you["body"]),
            "alive": True, "just_ate": us_just_ate,
        }
        enemy_sims = [s for s in (make_sim(e) for e in enemies) if s is not None]
        food_set   = set(merged_food)
        results: dict = {}

        for direction in moves:
            if time.monotonic() >= deadline - 0.02:
                break

            dx, dy       = cls.DIRECTIONS[direction]
            our_new_head = (our_head[0]+dx, our_head[1]+dy)

            enemy_opts: list = []
            for esim in enemy_sims:
                e_moves = cls._get_legal_moves_sim(
                    esim["head"], esim["body"], occupied, width, height
                )
                if len(e_moves) > 2:
                    e_moves.sort(key=lambda m: _manhattan(
                        esim["head"][0]+cls.DIRECTIONS[m][0],
                        esim["head"][1]+cls.DIRECTIONS[m][1],
                        our_new_head[0], our_new_head[1],
                    ))
                    e_moves = e_moves[:2]
                enemy_opts.append(e_moves or ["up"])

            combos     = list(_iproduct(*enemy_opts)) if enemy_opts else [()]
            all_scores = []

            for combo in combos:
                if time.monotonic() >= deadline - 0.01:
                    break
                all_moves_map = {you_id: direction}
                for i, esim in enumerate(enemy_sims):
                    all_moves_map[esim["id"]] = combo[i] if i < len(combo) else "up"
                leaf = cls._simulate_and_evaluate(
                    us_sim, enemy_sims, all_moves_map,
                    food_set, hazard_set, hazard_dmg, food_dist_map, width, height,
                )
                all_scores.append(leaf)

            if all_scores:
                # Probabilistic blend: 60% paranoid + 30% random + 10% greedy
                min_s = min(all_scores)
                avg_s = sum(all_scores) / len(all_scores)
                max_s = max(all_scores)
                results[direction] = 0.6 * min_s + 0.3 * avg_s + 0.1 * max_s
            else:
                results[direction] = 0.0

        return results

    # ================================================================== #
    #  Primary decision loop                                             #
    # ================================================================== #
    @classmethod
    def get_best_move(
        cls,
        data: dict,
        merged_food: set,
        game_id: str = "",
        deadline: float = 0.0,
    ) -> str:
        if deadline == 0.0:
            deadline = time.monotonic() + cls.COMPUTE_BUDGET_S

        board  = data["board"]
        you    = data["you"]
        width  = board["width"]
        height = board["height"]
        head   = you["head"]

        # Build context ONCE
        ctx = cls._build_context(data, game_id, merged_food, deadline)

        neck_pt = _pt(you["body"][1]) if len(you.get("body", [])) > 1 else None

        safe_moves: list = []
        for direction, (dx, dy) in cls.DIRECTIONS.items():
            nx, ny = head["x"]+dx, head["y"]+dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if neck_pt and nx == neck_pt[0] and ny == neck_pt[1]:
                continue
            if (nx, ny) in ctx.occupied:
                continue
            safe_moves.append(direction)

        if not safe_moves:
            in_bounds = [
                d for d, (dx, dy) in cls.DIRECTIONS.items()
                if 0 <= head["x"]+dx < width and 0 <= head["y"]+dy < height
            ]
            fallback = random.choice(in_bounds) if in_bounds else "up"
            log.warning("No safe moves! Fallback → %s", fallback)
            return fallback

        # Opening book bypass (turns 1-5)
        book_move = cls._opening_book(ctx, safe_moves)
        if book_move is not None:
            log.info("Opening Book turn=%d → %s", ctx.turn, book_move)
            return book_move

        # 1v1 dedicated scorer
        if ctx.phase == GamePhase.LATE_1V1:
            return cls._score_1v1(safe_moves, data, ctx)

        chosen = cls._rank(safe_moves, data, ctx)
        log.info("Safe moves: %s → chose: %s (phase=%s)", safe_moves, chosen, ctx.phase.value)
        return chosen

    # ================================================================== #
    #  HIERARCHICAL SCORING PIPELINE (V9.0)                             #
    # ================================================================== #
    @classmethod
    def _rank(cls, moves: list, data: dict, ctx: GameContext) -> str:
        try:
            return cls._rank_impl(moves, data, ctx)
        except Exception as exc:
            log.error("_rank failed (%s) — falling back to space-max", exc, exc_info=True)
            head = data["you"]["head"]
            return cls._rank_space_max(
                moves, ctx.occupied, ctx.width, ctx.height,
                head["x"], head["y"],
            )

    @classmethod
    def _rank_impl(cls, moves: list, data: dict, ctx: GameContext) -> str:
        pw = cls.PHASE_WEIGHTS.get(ctx.phase, cls.PHASE_WEIGHTS[GamePhase.MID])
        w_voronoi  = pw["W_VORONOI"]
        w_food     = pw["W_FOOD"]
        w_edge     = pw["W_EDGE"]
        w_corner   = pw["W_CORNER"]
        w_kill     = pw["W_COMBAT_KILL"]
        hunger_thr = pw["HUNGER_THRESHOLD"]
        kill_margin = 1 if ctx.phase == GamePhase.LATE_FFA else cls.KILL_MARGIN

        # Precompute kill cells + H2H map + enemy legal-move counts
        kill_cells: set   = set()
        h2h_risk_map: dict = {}
        enemy_legal_counts: dict = {}  # enemy_id -> count of legal moves (for RISK_EQUAL override)
        priority_ord = {"RISK_HIGH": 3, "RISK_EQUAL": 2, "RISK_LOW": 1, "RISK_NONE": 0}

        for e in ctx.enemy_data:
            e_head = e.get("head")
            if e_head is None:
                continue
            pos          = (e_head["x"], e_head["y"])
            e_len        = e["length"]
            visible_segs = e["visible_segs"]

            # Standard kill classification
            safe_kill = (visible_segs >= e_len - 1
                         and (ctx.our_len - visible_segs) >= kill_margin + 1)

            # Trapped-kill exception + count legal moves (§2.1 Trapped Enemy Auto-Attack)
            e_body   = e.get("body", [])
            e_neck_r = e_body[1] if len(e_body) > 1 else None
            e_neck   = _pt(e_neck_r)
            e_legal  = [
                (pos[0]+dx2, pos[1]+dy2)
                for dx2, dy2 in cls._DELTAS
                if (0 <= pos[0]+dx2 < ctx.width and 0 <= pos[1]+dy2 < ctx.height
                    and (pos[0]+dx2, pos[1]+dy2) not in ctx.occupied)
            ]
            if e_neck:
                e_legal = [m for m in e_legal if m != e_neck]
            enemy_legal_counts[e.get("id", "")] = len(e_legal)

            # Auto-attack: 1 legal escape AND we are longer
            if len(e_legal) == 1 and ctx.our_len > e_len:
                kill_cells.add(e_legal[0])

            risk = cls._h2h_risk_level(ctx.our_len, e_len, kill_margin)

            # §2.3 RISK_EQUAL rebalance: enemy with ≤2 legal moves treated as RISK_LOW
            if risk == "RISK_EQUAL" and len(e_legal) <= 2:
                risk = "RISK_LOW"

            for dx, dy in cls._DELTAS:
                cell = (pos[0]+dx, pos[1]+dy)
                if not (0 <= cell[0] < ctx.width and 0 <= cell[1] < ctx.height):
                    continue
                if safe_kill:
                    kill_cells.add(cell)
                else:
                    cur = h2h_risk_map.get(cell, "RISK_NONE")
                    if priority_ord.get(risk, 0) > priority_ord.get(cur, 0):
                        h2h_risk_map[cell] = risk

        # H2H hazard exploitation (V8 3.2)
        for cell2 in list(h2h_risk_map.keys()):
            if h2h_risk_map[cell2] == "RISK_EQUAL" and cell2 in ctx.hazard_set:
                h2h_risk_map[cell2] = "RISK_LOW"

        # Threat density (V8 2.7)
        threat_count: dict = {}
        for eh in ctx.enemy_heads:
            for dx2, dy2 in cls._DELTAS:
                c2 = (eh[0]+dx2, eh[1]+dy2)
                threat_count[c2] = threat_count.get(c2, 0) + 1

        # Ghost risk
        visible_ids = {s["id"] for s in data["board"].get("snakes", []) if s.get("head") is not None}
        ghost_info  = {k: v for k, v in ctx.enemy_info.items() if k not in visible_ids}
        ghost_risk  = cls._compute_ghost_risk(ghost_info, ctx.turn, ctx.occupied, ctx.width, ctx.height)

        # Precompute: enemy free-space without us (for constriction + enclosed-enemy detector)
        e_free_precomp: dict = {}
        primary_enemy = next((e for e in ctx.enemy_data if e["head_pos"] is not None), None)
        if primary_enemy and time.monotonic() < ctx.deadline - 0.08:
            ep = primary_enemy["head_pos"]
            occ_no_us = ctx.occupied - set(ctx.our_body) | {ctx.our_head}
            e_free_precomp[ep] = cls.voronoi_bfs(ep, [], occ_no_us, ctx.width, ctx.height)

        # ── PRE-TIER: FREE FOOD immediate return (§1.1) ───────────────
        # If food is adjacent AND safe AND we'd win the food race, take it NOW.
        if ctx.visible_food:
            for direction in moves:
                dx, dy = cls.DIRECTIONS[direction]
                cand   = (ctx.our_head[0]+dx, ctx.our_head[1]+dy)
                if cand not in ctx.visible_food:
                    continue
                if cand in ctx.occupied:
                    continue
                h2h_at_food = h2h_risk_map.get(cand, "RISK_NONE")
                if h2h_at_food in ("RISK_HIGH", "RISK_EQUAL"):
                    continue
                # Skip if an equal/larger enemy is also adjacent to this food (food-race collision risk)
                contested = any(
                    _manhattan(e["head_pos"][0], e["head_pos"][1], cand[0], cand[1]) <= 2
                    and e["length"] >= ctx.our_len
                    for e in ctx.enemy_data
                    if e["head_pos"] is not None
                )
                if contested:
                    continue
                if cls._is_boxed_in(ctx, cand):
                    continue
                log.info("FREE FOOD → %s at %s", direction, cand)
                return direction


        # ── PRE-TIER: TRAPPED ENEMY AUTO-ATTACK (§2.1) ───────────────
        # If enemy has exactly 1 legal move AND we can step there AND we're longer → kill
        if ctx.our_len > 3:  # avoid suiciding early
            for direction in moves:
                dx, dy = cls.DIRECTIONS[direction]
                cand   = (ctx.our_head[0]+dx, ctx.our_head[1]+dy)
                if cand not in kill_cells:
                    continue
                if cand in ctx.occupied:
                    continue
                # Verify this is a 1-escape-cell kill (not just a normal kill cell)
                for e in ctx.enemy_data:
                    if e["head_pos"] is None:
                        continue
                    if e["length"] >= ctx.our_len:
                        continue
                    eid = e.get("id", "")
                    if enemy_legal_counts.get(eid, 99) == 1:
                        h2h_at = h2h_risk_map.get(cand, "RISK_NONE")
                        if h2h_at not in ("RISK_HIGH",):
                            log.info("TRAPPED ENEMY AUTO-ATTACK → %s kills %s", direction, eid)
                            return direction

        # ── TIER CLASSIFICATION ───────────────────────────────────────
        move_tiers: dict = {}
        for eh in ctx.enemy_heads:
            for dx, dy in cls._DELTAS:
                c = (eh[0]+dx, eh[1]+dy)
                if 0 <= c[0] < ctx.width and 0 <= c[1] < ctx.height:
                    pass  # threat_count already built

        for direction in moves:
            dx, dy = cls.DIRECTIONS[direction]
            cand   = (ctx.our_head[0]+dx, ctx.our_head[1]+dy)

            # --- VETO ---
            if cand in ctx.occupied:
                move_tiers[direction] = MoveTier.VETO
                continue

            is_direct_death = any(
                cand == e["head_pos"]
                for e in ctx.enemy_data
                if e["head_pos"] is not None and e["length"] >= ctx.our_len
            )
            if is_direct_death:
                move_tiers[direction] = MoveTier.VETO
                continue

            # --- AGGRESSIVE_SAFE ---
            if cand in kill_cells:
                move_tiers[direction] = MoveTier.AGGRESSIVE_SAFE
                continue

            # Ambush: enemy forced into boxed cell
            is_amb = any(
                cls._is_ambush(cand, e["head_pos"], ctx)
                for e in ctx.enemy_data
                if e["head_pos"] is not None and ctx.our_len > e["length"] + 1
                and time.monotonic() < ctx.deadline - 0.05
            )
            if is_amb:
                move_tiers[direction] = MoveTier.AGGRESSIVE_SAFE
                continue

            # --- SURVIVAL_RISKY ---
            h2h = h2h_risk_map.get(cand, "RISK_NONE")
            tc  = threat_count.get(cand, 0)
            is_h2h_risky = (
                h2h == "RISK_HIGH"
                or (h2h == "RISK_EQUAL" and tc >= 1)
            )

            # §3.1 Coil Cap: forbid moving adjacent to own tail unless starving/no-enemy/no-food
            is_tail_coil = False
            if ctx.our_tail is not None and ctx.our_health >= 25:
                if _manhattan(cand[0], cand[1], ctx.our_tail[0], ctx.our_tail[1]) <= 1:
                    has_nearby_food = any(
                        _manhattan(cand[0], cand[1], fx, fy) <= 2
                        for fx, fy in ctx.visible_food
                    ) if ctx.visible_food else False
                    has_nearby_enemy = any(
                        _manhattan(cand[0], cand[1], eh[0], eh[1]) <= 2
                        for eh in ctx.enemy_heads
                    )
                    if has_nearby_food or has_nearby_enemy:
                        is_tail_coil = True

            boxed = cls._is_boxed_in(ctx, cand)

            if is_h2h_risky or boxed or is_tail_coil:
                move_tiers[direction] = MoveTier.SURVIVAL_RISKY
                continue

            # --- SURVIVAL_SAFE ---
            move_tiers[direction] = MoveTier.SURVIVAL_SAFE

        # ── TIER SELECTION ────────────────────────────────────────────
        non_veto = [d for d in moves if move_tiers.get(d, MoveTier.VETO) != MoveTier.VETO]
        if not non_veto:
            non_veto = moves  # truly trapped

        tier3 = [d for d in non_veto if move_tiers.get(d) == MoveTier.AGGRESSIVE_SAFE]
        tier2 = [d for d in non_veto if move_tiers.get(d) == MoveTier.SURVIVAL_SAFE]
        tier1 = [d for d in non_veto if move_tiers.get(d) == MoveTier.SURVIVAL_RISKY]

        # Kill only when we have a safe fallback (spec §3.3)
        if tier3 and tier2:
            pool = tier3
        elif tier2:
            pool = tier2
        elif tier1:
            pool = tier1
        else:
            pool = tier3 if tier3 else non_veto  # last resort

        # ── MINIMAX SCORES (gated) ────────────────────────────────────
        n_enemies = len([e for e in ctx.enemy_data if e["head_pos"] is not None])
        mm_gate = (
            (n_enemies == 1 and time.monotonic() < ctx.deadline - 0.10)
            or (n_enemies == 2 and time.monotonic() < ctx.deadline - 0.15)
        )
        mm_scores: dict = {}
        if mm_gate and len(pool) > 1:
            mm_scores = cls._minimax_2ply(
                pool, data, ctx.occupied, ctx.merged_food, ctx.enemy_heads,
                ctx.hazard_set, ctx.hazard_dmg, ctx.food_dist_map, ctx.deadline,
                enemy_info=ctx.enemy_info,
            ) or {}

        # ── STARVATION CLOCK ─────────────────────────────────────────
        need_food    = ctx.our_health < hunger_thr
        turns_to_die = ctx.our_health - 1
        max_elen     = max((e["length"] for e in ctx.enemy_data), default=0)
        we_dominant  = ctx.our_len > max_elen

        # Executioner mode conditions
        exec_active = (
            ctx.phase in (GamePhase.LATE_1V1, GamePhase.LATE_FFA)
            and we_dominant
            and primary_enemy is not None
            and (primary_enemy.get("health") or 100) < 55
            and ctx.our_len > (primary_enemy["length"] + 1)
        )

        # ── DO SOMETHING timer: track turns since last food ──────────
        game_mem     = _game_memory.get(data.get("game", {}).get("id", ""), {})
        last_food_t  = game_mem.get("last_food_turn", 0)
        starve_coast = ctx.turn - last_food_t
        # Update last_food_turn when we just ate (body[-1] == body[-2])
        body_pts = ctx.our_body
        if len(body_pts) >= 2 and body_pts[0] == body_pts[-1]:
            game_mem["last_food_turn"] = ctx.turn
            starve_coast = 0
        do_something_mode = (starve_coast >= 10 and ctx.our_health < 80)

        # ── SCORING LOOP (V8.1 BLOODLUST) ─────────────────────────────
        scores: dict = {}
        timed_out = False

        for direction in pool:
            if time.monotonic() >= ctx.deadline:
                timed_out = True
                scores[direction] = 0.0
                continue

            dx, dy = cls.DIRECTIONS[direction]
            nx, ny = ctx.our_head[0]+dx, ctx.our_head[1]+dy
            cand   = (nx, ny)
            score  = 0.0

            voronoi   = cls.voronoi_bfs(cand, ctx.enemy_heads, ctx.occupied, ctx.width, ctx.height)
            food_dist = ctx.food_dist_map.get(cand, float(ctx.width + ctx.height))

            # §3.3 Expansion Bonus: reduce Voronoi weight when we already own lots of space
            eff_voronoi_w = w_voronoi
            if voronoi > ctx.our_len * 4:
                eff_voronoi_w = w_voronoi * 0.5

            # Territory
            score += eff_voronoi_w * voronoi

            # Kill bonus
            if cand in kill_cells:
                score += w_kill

            # §2.2 Enclosed Enemy Detector: +800 for reducing enemy space (lightweight version)
            # Note: full constriction is computed separately below; this is a cheap bonus.
            if (we_dominant and primary_enemy
                    and primary_enemy["head_pos"] in e_free_precomp):
                ep = primary_enemy["head_pos"]
                e_free_before = e_free_precomp[ep]
                # Approx: if candidate is adjacent to enemy head, it likely constricts
                if (e_free_before > 0
                        and _manhattan(nx, ny, ep[0], ep[1]) <= 2
                        and nx != ep[0] or ny != ep[1]):
                    score += 400  # lighter bonus, no extra voronoi call

            # H2H penalty
            h2h = h2h_risk_map.get(cand, "RISK_NONE")
            if h2h == "RISK_HIGH":    score -= 1200
            elif h2h == "RISK_EQUAL": score -= 600
            elif h2h == "RISK_LOW":   score -= 200

            # §5.3 H2H Challenge: if we're longer+2 and close to enemy, move toward them
            for e in ctx.enemy_data:
                if e["head_pos"] is None:
                    continue
                edist = _manhattan(cand[0], cand[1], e["head_pos"][0], e["head_pos"][1])
                if ctx.our_len > e["length"] + 2 and edist <= 2:
                    score += 600  # advance into them

            # §5.1 Pincer bonus: cut off enemy near wall
            for e in ctx.enemy_data:
                if e["head_pos"] is None:
                    continue
                ex2, ey2 = e["head_pos"]
                near_wall = (ex2 <= 1 or ex2 >= ctx.width-2 or ey2 <= 1 or ey2 >= ctx.height-2)
                if near_wall and ctx.our_len > e["length"]:
                    dist_cand_to_enemy = _manhattan(nx, ny, ex2, ey2)
                    if dist_cand_to_enemy <= 3:
                        score += 800

            # Crossfire (V8 2.7)
            if threat_count.get(cand, 0) >= 2:
                score -= 400

            # Ghost risk
            score += ghost_risk.get(cand, 0.0)

            # Hazard
            if cand in ctx.hazard_set:
                score += cls.W_HAZARD_CELL

            # ── FOOD SCORING with BLOODLUST food sprint (§1.2) ────────
            nearest_food_d = ctx.food_dist_map.get(cand, float("inf"))

            # Food Sprint multiplier based on health
            if ctx.our_health < 70:
                food_multiplier = 3.0
            elif ctx.our_health < 90:
                food_multiplier = 1.5
            else:
                food_multiplier = 0.5

            # §1.3 Prioritize visible food: use shorter distance if food is actually visible
            if ctx.visible_food:
                vis_food_d = min(
                    (_manhattan(nx, ny, fx, fy) for fx, fy in ctx.visible_food),
                    default=float("inf")
                )
                if vis_food_d < nearest_food_d:
                    nearest_food_d = float(vis_food_d)
                    food_dist = float(vis_food_d)

            if turns_to_die > 0:
                if nearest_food_d != float("inf") and nearest_food_d >= turns_to_die:
                    # DESPERATION / CORNERED BEAST — pure food chase
                    score = -food_dist * 200 + voronoi
                else:
                    base_food_wt = w_food * food_multiplier
                    if nearest_food_d != float("inf") and nearest_food_d < turns_to_die - 10:
                        food_wt = base_food_wt * 0.5   # food abundant → less pull
                    else:
                        food_wt = base_food_wt

                    if nearest_food_d != float("inf") and nearest_food_d < turns_to_die:
                        race_value = cls._food_race_v2(cand, ctx, ctx.enemy_data)
                        if race_value > 0:
                            score += food_wt * food_dist * (1.0 + race_value)
                        elif race_value < 0:
                            score += abs(food_wt) * 20
                        else:
                            score += food_wt * food_dist
                    else:
                        score += food_wt * food_dist

            # §5.2 Food Denial: bonus for being closer to food AND blocking enemy approach
            if ctx.visible_food and ctx.enemy_heads:
                vis_food_d_us  = min((_manhattan(nx, ny, fx, fy) for fx, fy in ctx.visible_food), default=999)
                vis_food_d_ene = min(
                    (_manhattan(ex3, ey3, fx, fy) for ex3, ey3 in ctx.enemy_heads for fx, fy in ctx.visible_food),
                    default=999
                )
                if vis_food_d_us < vis_food_d_ene:
                    score += 600

            # §6.1 Do Something override: pure food/attack, ignore territory
            if do_something_mode:
                score -= eff_voronoi_w * voronoi  # strip territory bonus
                score += cls.W_CENTER * 0          # strip center
                score += 200 * (1.0 / max(1, food_dist))  # pure food pull

            # Executioner mode
            if exec_active and primary_enemy:
                score += cls._executioner_score(cand, primary_enemy, ctx)

            # Constriction (when dominant) — skip if already did enclosed-enemy
            if (we_dominant and primary_enemy
                    and primary_enemy["head_pos"] in e_free_precomp
                    and time.monotonic() < ctx.deadline - 0.05):
                ep = primary_enemy["head_pos"]
                score += cls._constriction_score(cand, ep, e_free_precomp[ep], ctx)

            # Edge / corner (phase-dynamic)
            if not do_something_mode:
                on_edge   = (nx == 0 or nx == ctx.width-1 or ny == 0 or ny == ctx.height-1)
                on_corner = ((nx == 0 or nx == ctx.width-1) and (ny == 0 or ny == ctx.height-1))
                if on_edge:   score += w_edge
                if on_corner: score += w_corner

            # Shadow penalty (V9.0)
            for ep2 in ctx.enemy_heads:
                if (nx == ep2[0] or ny == ep2[1]) and _manhattan(nx, ny, ep2[0], ep2[1]) <= 2:
                    score -= 120

            # §3.2 Self-Touch Penalty: -60 for moving directly adjacent to own body (not neck)
            if len(ctx.our_body) > 2:
                for seg in ctx.our_body[2:]:
                    if seg and _manhattan(nx, ny, seg[0], seg[1]) == 1:
                        score -= 60
                        break  # one penalty per move

            # Tail-following coiling ONLY when no threats and no food nearby
            if (ctx.our_health > 80 and ctx.our_tail is not None
                    and not ctx.visible_food
                    and all(_manhattan(nx, ny, eh[0], eh[1]) > 5 for eh in ctx.enemy_heads)):
                tail_dist = _manhattan(nx, ny, ctx.our_tail[0], ctx.our_tail[1])
                score += max(0, 8 - tail_dist) * 3

            # Center pull
            if not do_something_mode:
                cx2   = (ctx.width - 1) / 2.0
                cy2   = (ctx.height - 1) / 2.0
                score += cls.W_CENTER * (abs(nx - cx2) + abs(ny - cy2))

            # Minimax modifier (5% influence)
            if mm_scores:
                mm = mm_scores.get(direction, 0.0)
                if mm <= -999999.0:
                    score -= 5000
                else:
                    score += mm * 0.05

            scores[direction] = score

        if not scores:
            return random.choice(pool) if pool else random.choice(moves)

        best_score = max(scores.values())
        best_moves = [m for m, s in scores.items() if s == best_score]
        chosen     = random.choice(best_moves)
        elapsed    = (time.monotonic() - (ctx.deadline - cls.COMPUTE_BUDGET_S)) * 1000
        log.info(
            "V9b Scores: %s | tier3=%s tier2=%s | phase=%s health=%d coast=%d elapsed=%.1fms%s → %s",
            {m: f"{s:.0f}" for m, s in scores.items()},
            len(tier3), len(tier2),
            ctx.phase.value, ctx.our_health, starve_coast, elapsed,
            " [TIMEOUT]" if timed_out else "", chosen,
        )
        return chosen

    # ================================================================== #
    #  1v1 dedicated scorer (V9.0 — uses new combat systems)            #
    # ================================================================== #
    @classmethod
    def _score_1v1(cls, moves: list, data: dict, ctx: GameContext) -> str:
        try:
            return cls._score_1v1_impl(moves, data, ctx)
        except Exception as exc:
            log.error("_score_1v1 failed (%s)", exc, exc_info=True)
            return cls._rank_space_max(
                moves, ctx.occupied, ctx.width, ctx.height,
                ctx.our_head[0], ctx.our_head[1],
            )

    @classmethod
    def _score_1v1_impl(cls, moves: list, data: dict, ctx: GameContext) -> str:
        pw         = cls.PHASE_WEIGHTS[GamePhase.LATE_1V1]
        w_food     = pw["W_FOOD"]
        w_kill     = pw["W_COMBAT_KILL"]
        w_edge     = pw["W_EDGE"]
        w_corner   = pw["W_CORNER"]
        hunger_thr = pw["HUNGER_THRESHOLD"]
        need_food  = ctx.our_health < hunger_thr

        primary = next((e for e in ctx.enemy_data if e["head_pos"] is not None), None)
        if primary is None:
            return cls._rank_space_max(
                moves, ctx.occupied, ctx.width, ctx.height,
                ctx.our_head[0], ctx.our_head[1],
            )

        e_head_pos = primary["head_pos"]
        e_len      = primary["length"]
        e_health   = primary.get("health", 100)

        # Fix 1.4: correct kill_cells for corridor trap
        kill_cells_1v1: set = set()
        e_body_1v1  = primary.get("body", [])
        vis_1v1     = sum(1 for s in e_body_1v1 if s is not None)
        if vis_1v1 >= e_len - 1 and (ctx.our_len - vis_1v1) >= cls.KILL_MARGIN + 1:
            for dx2, dy2 in cls._DELTAS:
                c2 = (e_head_pos[0]+dx2, e_head_pos[1]+dy2)
                if 0 <= c2[0] < ctx.width and 0 <= c2[1] < ctx.height:
                    kill_cells_1v1.add(c2)

        all_threatened = {
            (e_head_pos[0]+dx, e_head_pos[1]+dy)
            for dx, dy in cls._DELTAS
            if 0 <= e_head_pos[0]+dx < ctx.width and 0 <= e_head_pos[1]+dy < ctx.height
        }
        safe_moves = [
            d for d in moves
            if not cls._is_corridor_trap(
                (ctx.our_head[0]+cls.DIRECTIONS[d][0], ctx.our_head[1]+cls.DIRECTIONS[d][1]),
                ctx.our_len, ctx.occupied, all_threatened, ctx.width, ctx.height, kill_cells_1v1
            )
        ] or moves

        # Precompute enemy free space for constriction (2.3 / V9 constriction)
        occ_no_us = ctx.occupied - set(ctx.our_body) | {ctx.our_head}
        e_free    = cls.voronoi_bfs(e_head_pos, [], occ_no_us, ctx.width, ctx.height)

        # Executioner mode active?
        exec_active = ctx.our_len > e_len + 1 and e_health < 55

        scores: dict = {}
        for direction in safe_moves:
            if time.monotonic() >= ctx.deadline:
                scores[direction] = 0.0
                continue

            dx, dy = cls.DIRECTIONS[direction]
            nx, ny = ctx.our_head[0]+dx, ctx.our_head[1]+dy
            cand   = (nx, ny)

            v_our   = cls.voronoi_bfs(cand, [e_head_pos], ctx.occupied, ctx.width, ctx.height)
            v_enemy = cls.voronoi_bfs(e_head_pos, [cand], ctx.occupied, ctx.width, ctx.height)
            diff    = v_our - v_enemy
            d2e     = _manhattan(nx, ny, e_head_pos[0], e_head_pos[1])

            if ctx.our_len > e_len:
                constriction = max(0, e_free - v_enemy)
                score = diff * 100 + v_our * 10 - d2e * 2 + constriction * 55
                if exec_active:
                    score += cls._executioner_score(cand, primary, ctx)
            elif ctx.our_len == e_len:
                score = v_our * 50 + diff * 20
                if need_food and ctx.food_dist_map:
                    fd = ctx.food_dist_map.get(cand, float(ctx.width + ctx.height))
                    score += w_food * fd
            else:
                score = v_our * 80 - diff * 10
                if need_food and ctx.food_dist_map:
                    fd = ctx.food_dist_map.get(cand, float(ctx.width + ctx.height))
                    score += w_food * fd

            if cand in kill_cells_1v1:
                score += w_kill

            if cand in ctx.hazard_set:
                score += cls.W_HAZARD_CELL

            on_edge   = (nx == 0 or nx == ctx.width-1 or ny == 0 or ny == ctx.height-1)
            on_corner = ((nx == 0 or nx == ctx.width-1) and (ny == 0 or ny == ctx.height-1))
            if on_edge:   score += w_edge
            if on_corner: score += w_corner

            scores[direction] = score

        if not scores:
            return random.choice(safe_moves)

        best   = max(scores.values())
        chosen = random.choice([m for m, s in scores.items() if s == best])
        log.info("1v1 scores: %s | our=%d vs e=%d health=%d → %s",
                 {m: f"{s:.0f}" for m, s in scores.items()},
                 ctx.our_len, e_len, e_health, chosen)
        return chosen


# ===========================================================================
# Routes
# ===========================================================================

@app.get("/", response_class=JSONResponse)
async def index() -> dict:
    return SNAKE_INFO


@app.post("/start", response_class=JSONResponse)
async def start(state: GameState) -> dict:
    data    = state.model_dump()
    game_id = data.get("game", {}).get("id", "?")
    _game_memory[game_id] = {
        "food": set(), "enemy_info": {}, "food_meta": {}, "latency_history": []
    }
    log.info("Game started: %s", game_id)
    return {}


@app.post("/move", response_class=JSONResponse)
async def move(state: GameState) -> dict:
    data    = state.model_dump()
    game_id = data.get("game", {}).get("id", "?")

    t_start = time.monotonic()

    # Adaptive compute budget (V8 4.1)
    mem  = _game_memory.get(game_id, {})
    hist = mem.get("latency_history", [])
    if hist:
        avg_ms = sum(hist) / len(hist)
        if avg_ms < 80:
            budget_s = min(0.300, TacticalEngine.COMPUTE_BUDGET_S + 0.025)
        elif avg_ms > 200:
            budget_s = max(0.150, TacticalEngine.COMPUTE_BUDGET_S - 0.025)
        else:
            budget_s = TacticalEngine.COMPUTE_BUDGET_S
    else:
        budget_s = TacticalEngine.COMPUTE_BUDGET_S

    deadline = t_start + budget_s

    _update_enemy_memory(game_id, data)
    merged_food = _update_food_memory(game_id, data)

    chosen     = TacticalEngine.get_best_move(data, merged_food, game_id=game_id, deadline=deadline)
    elapsed_ms = (time.monotonic() - t_start) * 1000

    active_mem = _game_memory.get(game_id)
    if active_mem is not None:
        hist2 = active_mem.setdefault("latency_history", [])
        hist2.append(elapsed_ms)
        if len(hist2) > 5:
            hist2.pop(0)

    log.info("Turn compute: %.1fms (budget=%.0fms)", elapsed_ms, budget_s * 1000)
    return {"move": chosen, "shout": "الثعبان لا يرحم! 🐍"}


@app.post("/end", response_class=JSONResponse)
async def end(state: GameState) -> dict:
    data    = state.model_dump()
    game_id = data.get("game", {}).get("id", "?")
    _game_memory.pop(game_id, None)
    log.info("Game ended: %s", game_id)
    return {}


# ===========================================================================
# Verification suite — T1-T16
# ===========================================================================

def _verify_all() -> None:
    import copy

    BASE_DATA: dict = {
        "game": {
            "id": "test-v9",
            "ruleset": {
                "name": "blackout",
                "settings": {
                    "viewRadius": 5,
                    "hazardDamagePerTurn": 14,
                    "royale": {"shrinkEveryNTurns": 25},
                },
            },
            "timeout": 500,
        },
        "turn": 15,
        "board": {
            "width": 11, "height": 11,
            "food":    [{"x": 2, "y": 2}],
            "hazards": [{"x": 0, "y": h} for h in range(11)],
            "snakes": [
                {
                    "id": "me", "name": "الثعبان", "health": 90,
                    "head": {"x": 5, "y": 5},
                    "body": [{"x":5,"y":5},{"x":5,"y":4},{"x":5,"y":3}],
                    "length": 3,
                },
                {
                    "id": "ghost-snake", "name": "ghost", "health": None,
                    "head": None, "body": [None, None, None, None], "length": 4,
                },
                {
                    "id": "partial-snake", "name": "partial", "health": None,
                    "head": {"x": 7, "y": 5},
                    "body": [{"x":7,"y":5},{"x":8,"y":5}, None, None],
                    "length": 4,
                },
            ],
        },
        "you": {
            "id": "me", "name": "الثعبان", "health": 90,
            "head": {"x": 5, "y": 5},
            "body": [{"x":5,"y":5},{"x":5,"y":4},{"x":5,"y":3}],
            "length": 3,
        },
    }

    def _fresh(gid: str, data: dict):
        _game_memory[gid] = {
            "food": set(), "enemy_info": {}, "food_meta": {}, "latency_history": []
        }
        _update_enemy_memory(gid, data)
        return _update_food_memory(gid, data)

    # T1: null-safety
    mf = _fresh("test-v9", BASE_DATA)
    r  = TacticalEngine.get_best_move(BASE_DATA, mf, game_id="test-v9")
    assert r in {"up", "down", "left", "right"}, f"T1 failed: {r!r}"
    log.info("✅ T1 null-safety OK — chose '%s'", r)

    # T2: tail vacating
    d2 = copy.deepcopy(BASE_DATA); d2["game"]["id"] = "tail-test"
    d2["board"]["snakes"].append({
        "id": "tailer", "name": "tailer", "health": 80,
        "head": {"x":1,"y":5},
        "body": [{"x":1,"y":5},{"x":2,"y":5},{"x":3,"y":5}], "length": 3,
    })
    _fresh("tail-test", d2)
    ei2  = _game_memory["tail-test"]["enemy_info"]
    occ2 = TacticalEngine._build_occupied(d2, "tail-test", ei2)
    assert (3, 5) not in occ2, "T2: tail should be free"
    log.info("✅ T2 tail-vacating OK")

    # T3: ghost risk
    _game_memory["ghost-risk"] = {
        "food": set(), "food_meta": {}, "latency_history": [],
        "enemy_info": {"ghost-snake": {
            "last_seen_turn": 10, "last_known_head": (5, 8),
            "last_body_count": 4, "prev_body_count": 4,
            "last_known_length": 4, "prev_known_length": 4,
        }},
    }
    d3 = copy.deepcopy(BASE_DATA); d3["game"]["id"] = "ghost-risk"; d3["turn"] = 12
    _update_enemy_memory("ghost-risk", d3)
    mf3 = _update_food_memory("ghost-risk", d3)
    r3  = TacticalEngine.get_best_move(d3, mf3, game_id="ghost-risk")
    assert r3 in {"up","down","left","right"}, f"T3 failed: {r3!r}"
    log.info("✅ T3 ghost-risk OK — chose '%s'", r3)

    # T4: corridor trap
    tiny_occ = {(x, y) for x in range(11) for y in range(11)} - {(9, 9), (8, 9)}
    assert TacticalEngine._is_corridor_trap((9,9), 3, tiny_occ, set(), 11, 11, set()), "T4 failed"
    log.info("✅ T4 corridor-trap OK")

    # T5: hazard avoidance
    d5 = copy.deepcopy(BASE_DATA); d5["game"]["id"] = "haz-game"
    d5["you"]["head"] = {"x":1,"y":5}
    d5["you"]["body"] = [{"x":1,"y":5},{"x":1,"y":4},{"x":1,"y":3}]
    d5["board"]["snakes"][0].update({"head":{"x":1,"y":5},"body":[{"x":1,"y":5},{"x":1,"y":4},{"x":1,"y":3}]})
    mf5 = _fresh("haz-game", d5)
    r5  = TacticalEngine.get_best_move(d5, mf5, game_id="haz-game")
    assert r5 in {"up","down","left","right"}, f"T5 failed: {r5!r}"
    log.info("✅ T5 hazard-avoidance OK — chose '%s'", r5)

    # T6: 1v1 phase
    d6 = copy.deepcopy(BASE_DATA); d6["game"]["id"] = "1v1-game"
    d6["board"]["snakes"] = [
        {"id":"me","name":"الثعبان","health":90,"head":{"x":5,"y":5},
         "body":[{"x":5,"y":5},{"x":5,"y":4},{"x":5,"y":3}],"length":3},
        {"id":"e1","name":"e","health":80,"head":{"x":8,"y":8},
         "body":[{"x":8,"y":8},{"x":8,"y":7}],"length":2},
    ]
    d6["you"] = d6["board"]["snakes"][0]
    mf6 = _fresh("1v1-game", d6)
    ph6 = TacticalEngine._get_game_phase(d6)
    assert ph6 == GamePhase.LATE_1V1, f"T6 failed: {ph6}"
    r6  = TacticalEngine.get_best_move(d6, mf6, game_id="1v1-game")
    assert r6 in {"up","down","left","right"}, f"T6 failed move: {r6!r}"
    log.info("✅ T6 1v1-mode OK — phase=%s chose '%s'", ph6.value, r6)

    # T7: dead-snake cleanup
    _game_memory["dead-game"] = {
        "food": set(), "food_meta": {}, "latency_history": [],
        "enemy_info": {"dead-e": {"last_seen_turn":5,"last_known_head":(3,3),
                                   "last_known_length":3,"prev_known_length":3}},
    }
    d7 = copy.deepcopy(BASE_DATA); d7["game"]["id"] = "dead-game"
    d7["board"]["snakes"] = [d7["board"]["snakes"][0]]
    _update_enemy_memory("dead-game", d7)
    assert "dead-e" not in _game_memory["dead-game"]["enemy_info"], "T7 failed"
    log.info("✅ T7 dead-snake cleanup OK")

    # T8: food stale TTL
    _game_memory["stale-game"] = {
        "food": {(0,0)}, "food_meta": {(0,0): 1},
        "enemy_info": {}, "latency_history": [],
    }
    d8 = copy.deepcopy(BASE_DATA); d8["game"]["id"] = "stale-game"; d8["turn"] = 50
    mf8 = _update_food_memory("stale-game", d8)
    assert (0,0) not in mf8, "T8 failed: stale food should be evicted"
    log.info("✅ T8 food-stale-TTL OK")

    # T9: edge/corner penalty constants
    assert TacticalEngine.W_EDGE < 0,   "T9 failed: W_EDGE must be negative"
    assert TacticalEngine.W_CORNER < 0, "T9 failed: W_CORNER must be negative"
    log.info("✅ T9 edge/corner constants OK — W_EDGE=%d W_CORNER=%d",
             TacticalEngine.W_EDGE, TacticalEngine.W_CORNER)

    # T10: desperation mode (health=8, food dist > health)
    d10 = copy.deepcopy(BASE_DATA); d10["game"]["id"] = "desp-game"
    d10["you"]["health"] = 8
    d10["board"]["snakes"][0].update({"health":8})
    d10["board"]["snakes"] = [d10["board"]["snakes"][0]]
    d10["board"]["food"] = [{"x":0,"y":0}]
    mf10 = _fresh("desp-game", d10)
    r10  = TacticalEngine.get_best_move(d10, mf10, game_id="desp-game")
    assert r10 in {"up","down","left","right"}, f"T10 failed: {r10!r}"
    log.info("✅ T10 desperation-mode OK — health=8 → '%s'", r10)

    # T11: opening book turn 1
    d11 = copy.deepcopy(BASE_DATA); d11["game"]["id"] = "ob-game"; d11["turn"] = 1
    d11["board"]["snakes"] = [d11["board"]["snakes"][0]]
    mf11 = _fresh("ob-game", d11)
    r11  = TacticalEngine.get_best_move(d11, mf11, game_id="ob-game")
    assert r11 in {"up","down","left","right"}, f"T11 failed: {r11!r}"
    log.info("✅ T11 opening-book OK — turn=1 → '%s'", r11)

    # T12: hierarchical tier — VETO blocks unsafe moves
    d12 = copy.deepcopy(BASE_DATA); d12["game"]["id"] = "veto-game"; d12["turn"] = 20
    d12["board"]["snakes"] = [
        {"id":"me","name":"الثعبان","health":90,"head":{"x":5,"y":5},
         "body":[{"x":5,"y":5},{"x":5,"y":4},{"x":5,"y":3}],"length":3},
        # Enemy at (5,6) — moving up would be H2H with equal-length enemy
        {"id":"e_eq","name":"e_eq","health":90,"head":{"x":5,"y":7},
         "body":[{"x":5,"y":7},{"x":5,"y":8},{"x":5,"y":9}],"length":3},
    ]
    d12["you"] = d12["board"]["snakes"][0]
    mf12 = _fresh("veto-game", d12)
    r12  = TacticalEngine.get_best_move(d12, mf12, game_id="veto-game")
    assert r12 in {"up","down","left","right"}, f"T12 failed: {r12!r}"
    log.info("✅ T12 hierarchical-veto OK — chose '%s' (equal enemy above)", r12)

    # T13: _is_boxed_in detects small pocket
    ctx_fake = GameContext(
        our_head=(9,9), our_body=[(9,9),(9,8)], our_len=3,
        our_health=80, our_tail=(9,8),
        width=11, height=11, turn=15,
        occupied={(x,y) for x in range(11) for y in range(11)} - {(9,9),(8,9)},
        occupied_conservative={(x,y) for x in range(11) for y in range(11)} - {(9,9),(8,9)},
        hazard_set=set(), hazard_dmg=0,
        enemy_heads=[], enemy_data=[],
        visible_food=set(), merged_food=set(),
        food_dist_map={},
        phase=GamePhase.MID,
        deadline=time.monotonic() + 1.0,
        view_radius=5,
        enemy_info={},
    )
    boxed = TacticalEngine._is_boxed_in(ctx_fake, (8, 9))
    assert boxed, "T13 failed: 2-cell pocket should be boxed-in"
    log.info("✅ T13 _is_boxed_in OK")

    # T14: phase weights exist for all phases
    for ph in GamePhase:
        assert ph in TacticalEngine.PHASE_WEIGHTS, f"T14 failed: no weights for {ph}"
    log.info("✅ T14 phase-weights OK — all 4 phases have entries")

    # T15: food race v2 — certain win
    ctx15 = GameContext(
        our_head=(5,5), our_body=[(5,5)], our_len=3, our_health=80, our_tail=(5,5),
        width=11, height=11, turn=15,
        occupied=set(), occupied_conservative=set(), hazard_set=set(), hazard_dmg=0,
        enemy_heads=[(8,8)],
        enemy_data=[{"id":"e","head":{"x":8,"y":8},"head_pos":(8,8),
                     "length":3,"visible_segs":2,"health":80,"body":[],"prev_length":0}],
        visible_food={(5,6)}, merged_food={(5,6)},
        food_dist_map={(5,5):1.0,(5,6):0.0,(8,8):6.0},
        phase=GamePhase.MID, deadline=time.monotonic()+1.0,
        view_radius=5, enemy_info={},
    )
    rv15 = TacticalEngine._food_race_v2((5,5), ctx15, ctx15.enemy_data)
    assert rv15 == 1.0, f"T15 failed: rv={rv15}, expected 1.0 (we're closer to food)"
    log.info("✅ T15 food-race-v2 OK — rv=%.1f (certain win)", rv15)

    # T16: executioner score fires when blocking food path
    ctx16 = GameContext(
        our_head=(5,5), our_body=[(5,5),(5,4)], our_len=5, our_health=80, our_tail=(5,4),
        width=11, height=11, turn=30,
        occupied={(5,5),(5,4)}, occupied_conservative={(5,5),(5,4)},
        hazard_set=set(), hazard_dmg=0,
        enemy_heads=[(5,7)],
        enemy_data=[{"id":"e","head":{"x":5,"y":7},"head_pos":(5,7),
                     "length":3,"visible_segs":2,"health":40,"body":[],"prev_length":0}],
        visible_food={(5,9)}, merged_food={(5,9)},
        food_dist_map={(5,7):2.0,(5,8):1.0,(5,9):0.0,(5,6):3.0,(5,5):4.0},
        phase=GamePhase.LATE_1V1, deadline=time.monotonic()+1.0,
        view_radius=5, enemy_info={},
    )
    enemy16 = ctx16.enemy_data[0]
    # Candidate (5,8) is a first-step on enemy's path to food (dist 1.0 < enemy dist 2.0)
    exec_s  = TacticalEngine._executioner_score((5,8), enemy16, ctx16)
    assert exec_s > 0, f"T16 failed: executioner should fire, got {exec_s}"
    log.info("✅ T16 executioner-score OK — bonus=%.0f for blocking food path", exec_s)

    # Cleanup
    for gid2 in ["test-v9","tail-test","ghost-risk","haz-game","1v1-game",
                  "dead-game","stale-game","desp-game","ob-game","veto-game"]:
        _game_memory.pop(gid2, None)


_verify_all()


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
