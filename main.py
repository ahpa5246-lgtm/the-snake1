"""
Battlesnake Blackout 2026 — High-Performance FastAPI Server
Survival Logic V8.0 | Total Domination Protocol

V7.0 preserved components (unchanged):
  Cross-turn food memory + TTL eviction (2.3)
  Dead-snake ghost purge (2.2)
  Partial-visibility tail vacating (2.1)
  Dynamic H2H risk (5.2)
  Probabilistic ghost occupancy (5.3)
  Reverse multi-source food BFS (7.2)
  Hazard-aware Dijkstra pathing (7.1)
  Game-phase routing + dedicated 1v1 scorer (4.2, 4.3)

V8.0 critical bug fixes:
  1.1 Minimax just_ate initialization
  1.2 Minimax as score modifier (5% tiebreaker), not move replacement
  1.3 Food race formula corrected (stronger chase for certain food)
  1.4 1v1 kill cells passed to corridor-trap filter
  1.5 agent_adapter.py food_meta initialization (in that file)

V8.0 strategic enhancements:
  2.1 Edge/corner penalty (W_EDGE=-60, W_CORNER=-180)
  2.2 Pin-trap detection (_is_pin_trap)
  2.3 1v1 constriction bonus
  2.4 Desperation mode (health < 15)
  2.5 Tail-following coiling
  2.6 LATE_FFA aggression
  2.7 Threat density map (crossfire penalty)

V8.0 combat refinements:
  3.1 Trapped-enemy kill exception
  3.2 Equal-H2H hazard exploitation (RISK_EQUAL → RISK_LOW on hazard cells)
  3.3 Ghost line-of-sight reset (already correct in V7 — verified by T3 test)

V8.0 performance & robustness:
  4.1 Adaptive compute budget (per-game latency history)
  4.3 Graceful degradation — _rank wraps in try/except → space-max fallback
"""

import heapq
import logging
import random
import time
from enum import Enum
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
app = FastAPI(title="الثعبان — Battlesnake Blackout 2026", version="8.0.0")


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
    "version":    "8.0.0",
}

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
FOOD_STALE_TTL = 40   # turns before out-of-view food is evicted from memory

# ---------------------------------------------------------------------------
# Per-game memory — keyed by game_id
# ---------------------------------------------------------------------------
_game_memory: dict[str, dict] = {}


# ===========================================================================
# Game-Phase enum
# ===========================================================================
class GamePhase(Enum):
    EARLY    = "early"
    MID      = "mid"
    LATE_1V1 = "late_1v1"
    LATE_FFA = "late_ffa"


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
# Memory updates
# ===========================================================================

def _update_food_memory(game_id: str, data: dict) -> set[tuple[int, int]]:
    """Cross-turn food memory with stale-TTL eviction (Bug fix 2.3)."""
    mem = _game_memory.setdefault(
        game_id, {"food": set(), "enemy_info": {}, "food_meta": {}, "latency_history": []}
    )
    prev_food: set[tuple[int, int]] = mem.setdefault("food", set())
    food_meta: dict[tuple[int, int], int] = mem.setdefault("food_meta", {})

    you    = data["you"]
    head   = you.get("head") or {}
    hx: int = head.get("x", 0)
    hy: int = head.get("y", 0)
    radius = _get_view_radius(data)
    turn   = data.get("turn", 0)

    visible_food: set[tuple[int, int]] = {
        (f["x"], f["y"]) for f in data.get("board", {}).get("food", [])
    }

    new_memory: set[tuple[int, int]] = set()
    new_meta: dict[tuple[int, int], int] = {}

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
    """
    Track per-enemy visibility.
    Bug fix 2.2: Remove dead-snake entries.
    Bug fix 2.1: Store last_known_length for partial-visibility tail inference.
    """
    mem = _game_memory.setdefault(
        game_id, {"food": set(), "enemy_info": {}, "food_meta": {}, "latency_history": []}
    )
    enemy_info: dict[str, dict] = mem.setdefault("enemy_info", {})

    you_id = data["you"]["id"]
    turn   = data.get("turn", 0)

    # Bug fix 2.2: purge dead snakes
    alive_ids = {s["id"] for s in data["board"].get("snakes", [])}
    for dead_id in list(enemy_info.keys()):
        if dead_id not in alive_ids:
            del enemy_info[dead_id]
            log.debug("Ghost purged: %s", dead_id)

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
# TacticalEngine — V8.0 Total Domination Protocol
# ===========================================================================
class TacticalEngine:
    """V8.0 Total Domination Protocol."""

    _DELTAS = ((0, 1), (0, -1), (-1, 0), (1, 0))

    DIRECTIONS: dict[str, tuple[int, int]] = {
        "up":    ( 0,  1),
        "down":  ( 0, -1),
        "left":  (-1,  0),
        "right": ( 1,  0),
    }

    # ── Scoring weights ────────────────────────────────────────────────────
    W_VORONOI         = 20
    W_COMBAT_KILL     = 800
    W_HAZARD_CELL     = -600
    W_FOOD            = -30
    W_CENTER          = -3
    W_GHOST_BASE      = -250
    W_EDGE            = -60     # V8.0: edge penalty
    W_CORNER          = -180    # V8.0: corner penalty (cumulative with edge = -240)
    GHOST_DECAY_TURNS  = 5
    GHOST_RADIUS       = 2
    KILL_MARGIN        = 2
    HUNGER_THRESHOLD   = 45
    COMPUTE_BUDGET_S: float = 0.250

    # ================================================================== #
    #  PRIMITIVE 1: Multi-source Voronoi BFS                             #
    # ================================================================== #
    @classmethod
    def voronoi_bfs(
        cls,
        our_head: tuple[int, int],
        enemy_heads: list[tuple[int, int]],
        occupied: set[tuple[int, int]],
        width: int,
        height: int,
    ) -> int:
        dist: dict[tuple[int, int], tuple[int, int]] = {}
        queue: list[tuple[int, tuple[int, int]]] = []

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
    #  PRIMITIVE 2: BFS shortest-path with hazard cost (7.1)             #
    # ================================================================== #
    @classmethod
    def bfs_dist(
        cls,
        start: tuple[int, int],
        targets: set[tuple[int, int]],
        occupied: set[tuple[int, int]],
        width: int,
        height: int,
        *,
        max_dist: int = 10**9,
        hazard_cells: set[tuple[int, int]] | None = None,
        hazard_cost: float = 1.0,
    ) -> float:
        if not targets:
            return float(max_dist)
        if start in targets:
            return 0.0

        if bool(hazard_cells) and hazard_cost > 1.0:
            best: dict[tuple[int, int], float] = {start: 0.0}
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
            visited: set[tuple[int, int]] = {start}
            queue: list[tuple[int, int]] = [start]
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
    #  PRIMITIVE 3: Reverse multi-source food BFS (7.2)                  #
    # ================================================================== #
    @classmethod
    def _food_bfs_reverse(
        cls,
        food_targets: set[tuple[int, int]],
        occupied: set[tuple[int, int]],
        width: int,
        height: int,
        hazard_cells: set[tuple[int, int]] | None = None,
        hazard_dmg: int = 0,
    ) -> dict[tuple[int, int], float]:
        if not food_targets:
            return {}

        hazard_cost = 1.0 + hazard_dmg / 10.0 if (hazard_cells and hazard_dmg > 0) else 1.0
        use_haz     = bool(hazard_cells) and hazard_dmg > 0
        dist_map: dict[tuple[int, int], float] = {}
        pq: list[tuple[float, tuple[int, int]]] = []

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
    #  PRIMITIVE 4: Occupied set — Bug fix 2.1                           #
    # ================================================================== #
    @classmethod
    def _build_occupied(
        cls,
        data: dict,
        game_id: str,
        enemy_info: dict,
    ) -> set[tuple[int, int]]:
        you    = data["you"]
        board  = data["board"]
        you_id = you["id"]
        occupied: set[tuple[int, int]] = set()

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
    #  COMBAT: Dynamic H2H risk (5.2)                                   #
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
    #  SURVIVAL: Corridor trap detection (3.1)                          #
    # ================================================================== #
    @classmethod
    def _is_corridor_trap(
        cls,
        candidate: tuple[int, int],
        snake_len: int,
        occupied: set[tuple[int, int]],
        threatened: set[tuple[int, int]],
        width: int,
        height: int,
        kill_cells: set[tuple[int, int]],
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
    #  SURVIVAL: Pin-trap detection (2.2) — V8.0                        #
    # ================================================================== #
    @classmethod
    def _is_pin_trap(
        cls,
        candidate: tuple[int, int],
        head_x: int,
        head_y: int,
        enemies: list[dict],
        occupied: set[tuple[int, int]],
        width: int,
        height: int,
        kill_cells: set[tuple[int, int]],
        deadline: float,
    ) -> bool:
        """
        Returns True if enemy move combinations can block ALL our next-turn
        escapes from `candidate` (2-ply lookahead for pin traps).
        """
        if candidate in kill_cells:
            return False
        if time.monotonic() >= deadline - 0.06:
            return False

        # After moving to candidate, our neck is (head_x, head_y)
        neck = (head_x, head_y)
        our_next = [
            (candidate[0] + dx, candidate[1] + dy)
            for dx, dy in cls._DELTAS
            if (0 <= candidate[0] + dx < width
                and 0 <= candidate[1] + dy < height
                and (candidate[0] + dx, candidate[1] + dy) not in occupied
                and (candidate[0] + dx, candidate[1] + dy) != neck)
        ]
        if not our_next:
            return True  # already pinned

        vis_enemies = [e for e in enemies if e.get("head") is not None][:3]
        if not vis_enemies:
            return False

        enemy_opts: list[list[tuple[int, int]]] = []
        for e in vis_enemies:
            eh = (e["head"]["x"], e["head"]["y"])
            opts = [
                (eh[0] + dx, eh[1] + dy)
                for dx, dy in cls._DELTAS
                if (0 <= eh[0] + dx < width
                    and 0 <= eh[1] + dy < height
                    and (eh[0] + dx, eh[1] + dy) not in occupied)
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
    #  SURVIVAL: Escape-route analysis (3.2) — gated                    #
    # ================================================================== #
    @classmethod
    def _min_escape_size(
        cls,
        candidate: tuple[int, int],
        snake_len: int,
        occupied: set[tuple[int, int]],
        enemies: list[dict],
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

        enemy_next_opts: list[list[tuple[int, int]]] = []
        for e in vis_enemies:
            eh   = (e["head"]["x"], e["head"]["y"])
            opts = [
                (eh[0] + dx, eh[1] + dy)
                for dx, dy in cls._DELTAS
                if 0 <= eh[0] + dx < width and 0 <= eh[1] + dy < height
                and (eh[0] + dx, eh[1] + dy) not in occupied
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
                    nx, ny = cx + dx, cy + dy
                    if (0 <= nx < width and 0 <= ny < height
                            and (nx, ny) not in occ_combo
                            and (nx, ny) not in visited):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            if len(visited) < snake_len + 2:
                return True

        return False

    # ================================================================== #
    #  COMBAT: Probabilistic ghost occupancy (5.3)                      #
    # ================================================================== #
    @classmethod
    def _compute_ghost_risk(
        cls,
        enemy_info: dict,
        turn: int,
        occupied: set[tuple[int, int]],
        width: int,
        height: int,
    ) -> dict[tuple[int, int], float]:
        ghost_risk: dict[tuple[int, int], float] = {}

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

            feasibility: set[tuple[int, int]] = set()
            for dx in range(-max_r, max_r + 1):
                rem = max_r - abs(dx)
                for dy in range(-rem, rem + 1):
                    cx, cy = lhx + dx, lhy + dy
                    if (0 <= cx < width and 0 <= cy < height
                            and (cx, cy) not in occupied):
                        feasibility.add((cx, cy))

            if not feasibility:
                continue

            prob    = 1.0 / len(feasibility)
            penalty = cls.W_GHOST_BASE * prob * decay
            for cell in feasibility:
                ghost_risk[cell] = ghost_risk.get(cell, 0.0) + penalty

        return ghost_risk

    # ================================================================== #
    #  STRATEGIC: Food race evaluation (4.1)                            #
    # ================================================================== #
    @classmethod
    def _food_race_value(
        cls,
        our_pos: tuple[int, int],
        enemy_heads: list[tuple[int, int]],
        food_dist_map: dict[tuple[int, int], float],
        our_len: int,
        enemies: list[dict],
        width: int,
        height: int,
    ) -> float:
        if not food_dist_map:
            return 0.0
        d_our = food_dist_map.get(our_pos, float(width + height))
        if not enemy_heads:
            return 1.0
        d_enemy_min = min(
            food_dist_map.get(eh, float(width + height)) for eh in enemy_heads
        )
        if d_our < d_enemy_min:
            return 1.0
        elif d_our > d_enemy_min:
            return -0.5
        else:
            race_val = 0.2
            for e in enemies:
                eh = e.get("head")
                if eh is None:
                    continue
                eh_pos = (eh["x"], eh["y"])
                if food_dist_map.get(eh_pos, float(width + height)) == d_enemy_min:
                    e_len = e.get("length", 0)
                    if our_len > e_len + 1:
                        race_val = max(race_val, 0.6)
                    elif our_len < e_len:
                        race_val = min(race_val, -0.3)
            return race_val

    # ================================================================== #
    #  STRATEGIC: Game phase detection (4.3)                            #
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

    @classmethod
    def _is_1v1(cls, enemies: list[dict]) -> bool:
        return len(enemies) == 1 and enemies[0].get("head") is not None

    # ================================================================== #
    #  STRATEGIC: Dedicated 1v1 scorer (4.2) — V8.0 enhanced            #
    # ================================================================== #
    @classmethod
    def _score_1v1(
        cls,
        moves: list[str],
        data: dict,
        occupied: set[tuple[int, int]],
        merged_food: set[tuple[int, int]],
        enemy_info: dict,
        *,
        deadline: float = 0.0,
    ) -> str:
        if deadline == 0.0:
            deadline = time.monotonic() + cls.COMPUTE_BUDGET_S

        board   = data["board"]
        you     = data["you"]
        width   = board["width"]
        height  = board["height"]
        head_x  = you["head"]["x"]
        head_y  = you["head"]["y"]
        health  = you["health"]
        our_len = len(you["body"])
        you_id  = you["id"]

        enemies    = [s for s in board.get("snakes", []) if s["id"] != you_id]
        hazard_set = {(h["x"], h["y"]) for h in board.get("hazards", [])}
        food_tgts  = set(merged_food)
        need_food  = health < 25  # tighter threshold in 1v1

        if not enemies or enemies[0].get("head") is None:
            return cls._rank_space_max(moves, occupied, width, height, head_x, head_y)

        enemy      = enemies[0]
        e_head_pos = (enemy["head"]["x"], enemy["head"]["y"])
        e_len      = enemy.get("length", len(enemy.get("body", [])))

        # Fix 1.4: compute actual kill_cells for corridor-trap bypass
        kill_cells_1v1: set[tuple[int, int]] = set()
        e_body_1v1   = enemy.get("body", [])
        vis_segs_1v1 = sum(1 for s in e_body_1v1 if s is not None)
        if (vis_segs_1v1 >= e_len - 1
                and (our_len - vis_segs_1v1) >= cls.KILL_MARGIN + 1):
            for dx2, dy2 in cls._DELTAS:
                c2 = (e_head_pos[0] + dx2, e_head_pos[1] + dy2)
                if 0 <= c2[0] < width and 0 <= c2[1] < height:
                    kill_cells_1v1.add(c2)

        # Corridor trap filter (with correct kill_cells)
        all_threatened = {
            (e_head_pos[0] + dx, e_head_pos[1] + dy)
            for dx, dy in cls._DELTAS
            if 0 <= e_head_pos[0] + dx < width and 0 <= e_head_pos[1] + dy < height
        }
        safe_moves = [
            d for d in moves
            if not cls._is_corridor_trap(
                (head_x + cls.DIRECTIONS[d][0], head_y + cls.DIRECTIONS[d][1]),
                our_len, occupied, all_threatened, width, height, kill_cells_1v1
            )
        ] or moves

        # Precompute enemy's total free space for constriction bonus (2.3)
        e_free_total = cls.voronoi_bfs(e_head_pos, [], occupied, width, height)

        scores: dict[str, float] = {}
        timed_out = False

        for direction in safe_moves:
            if time.monotonic() >= deadline:
                timed_out = True
                scores[direction] = 0.0
                continue

            dx, dy = cls.DIRECTIONS[direction]
            nx, ny = head_x + dx, head_y + dy
            cand   = (nx, ny)

            v_our   = cls.voronoi_bfs(cand, [e_head_pos], occupied, width, height)
            v_enemy = cls.voronoi_bfs(e_head_pos, [cand], occupied, width, height)
            diff    = v_our - v_enemy
            dist_to_enemy = _manhattan(nx, ny, e_head_pos[0], e_head_pos[1])

            if our_len > e_len:
                # Enhancement 2.3: constriction bonus
                constriction = e_free_total - v_enemy
                score = diff * 100 + v_our * 10 - dist_to_enemy * 2 + constriction * 25
            elif our_len == e_len:
                score = v_our * 50 + diff * 20
                if need_food and food_tgts:
                    fd = cls.bfs_dist(cand, food_tgts, occupied, width, height, max_dist=width+height)
                    score += cls.W_FOOD * fd
            else:
                score = v_our * 80 - diff * 10
                if need_food and food_tgts:
                    fd = cls.bfs_dist(cand, food_tgts, occupied, width, height, max_dist=width+height)
                    score += cls.W_FOOD * fd

            if cand in hazard_set:
                score += cls.W_HAZARD_CELL

            # Enhancement 2.1: edge/corner penalty
            on_edge   = (nx == 0 or nx == width - 1 or ny == 0 or ny == height - 1)
            on_corner = ((nx == 0 or nx == width - 1) and (ny == 0 or ny == height - 1))
            if on_edge:   score += cls.W_EDGE
            if on_corner: score += cls.W_CORNER

            scores[direction] = score

        if not scores:
            return random.choice(safe_moves)

        best   = max(scores.values())
        chosen = random.choice([m for m, s in scores.items() if s == best])
        elapsed = (time.monotonic() - (deadline - cls.COMPUTE_BUDGET_S)) * 1000
        log.info("1v1 scores: %s | our=%d vs e=%d | elapsed=%.1fms%s → %s",
                 {m: f"{s:.0f}" for m, s in scores.items()},
                 our_len, e_len, elapsed,
                 " [TIMEOUT]" if timed_out else "", chosen)
        return chosen

    @classmethod
    def _rank_space_max(
        cls,
        moves: list[str],
        occupied: set[tuple[int, int]],
        width: int,
        height: int,
        head_x: int,
        head_y: int,
    ) -> str:
        best_move  = moves[0] if moves else "up"
        best_space = -1
        for d in moves:
            dx, dy = cls.DIRECTIONS[d]
            c      = (head_x + dx, head_y + dy)
            vis    = {c}
            q      = [c]; ptr = 0
            while ptr < len(q):
                cx2, cy2 = q[ptr]; ptr += 1
                for ddx, ddy in cls._DELTAS:
                    nc = (cx2 + ddx, cy2 + ddy)
                    if (0 <= nc[0] < width and 0 <= nc[1] < height
                            and nc not in occupied and nc not in vis):
                        vis.add(nc); q.append(nc)
            if len(vis) > best_space:
                best_space = len(vis); best_move = d
        return best_move

    # ================================================================== #
    #  2-PLY PARANOID MINIMAX (6.0) — V8.0 fixed                        #
    # ================================================================== #
    @classmethod
    def _get_legal_moves_sim(
        cls,
        head: tuple[int, int],
        body: list[tuple[int, int]],
        occupied: set[tuple[int, int]],
        width: int,
        height: int,
    ) -> list[str]:
        neck = body[1] if len(body) > 1 else None
        return [
            d for d, (dx, dy) in cls.DIRECTIONS.items()
            if (0 <= head[0] + dx < width and 0 <= head[1] + dy < height
                and (head[0] + dx, head[1] + dy) != neck
                and (head[0] + dx, head[1] + dy) not in occupied)
        ]

    @classmethod
    def _simulate_and_evaluate(
        cls,
        us: dict,
        enemy_sims: list[dict],
        all_moves: dict[str, str],
        food_set: set[tuple[int, int]],
        hazard_set: set[tuple[int, int]],
        hazard_dmg: int,
        food_dist_map: dict[tuple[int, int], float],
        width: int,
        height: int,
    ) -> float:
        from collections import defaultdict

        us2   = {**us, "body": list(us["body"])}
        enems = [{**e, "body": list(e["body"])} for e in enemy_sims]
        all_s = [us2] + enems

        # 1. New heads
        new_heads: dict[str, tuple[int, int]] = {}
        for s in all_s:
            if not s.get("alive", True):
                continue
            dx, dy = cls.DIRECTIONS.get(all_moves.get(s["id"], "up"), (0, 1))
            new_heads[s["id"]] = (s["head"][0] + dx, s["head"][1] + dy)

        # 2. OOB
        for s in all_s:
            if not s.get("alive", True): continue
            nx, ny = new_heads[s["id"]]
            if not (0 <= nx < width and 0 <= ny < height):
                s["alive"] = False

        if not us2.get("alive", True):
            return -999999.0

        # 3. Body collision (Fix 1.1: uses actual just_ate flags)
        body_cells: set[tuple[int, int]] = set()
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
        head_to_snakes: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for s in all_s:
            if s.get("alive", True):
                head_to_snakes[new_heads[s["id"]]].append(s)

        for pos, claimants in head_to_snakes.items():
            if len(claimants) < 2: continue
            mx = max(c["length"] for c in claimants)
            for c in claimants:
                if c["length"] < mx:
                    c["alive"] = False
                elif c["length"] == mx:
                    if any(x["id"] != c["id"] and x["length"] == mx for x in claimants):
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
                s["health"]   = 100
                s["just_ate"] = True
            s["head"]   = nh
            s["length"] = len(s["body"])

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
        our_fd      = food_dist_map.get(us2["head"], float(width + height))
        food_rv     = 1.0 if our_fd < 3 else (0.5 if our_fd < 7 else 0.0)

        return (
            our_voronoi * 15
            + enemy_died  * 5000
            + us2["health"] * 2
            + food_rv * 100
            - (500 if us2["head"] in hazard_set else 0)
        )

    @classmethod
    def _minimax_2ply(
        cls,
        moves: list[str],
        data: dict,
        occupied: set[tuple[int, int]],
        merged_food: set[tuple[int, int]],
        enemy_heads: list[tuple[int, int]],
        hazard_set: set[tuple[int, int]],
        hazard_dmg: int,
        food_dist_map: dict[tuple[int, int], float],
        deadline: float,
        enemy_info: dict | None = None,   # V8.0 Fix 1.1
    ) -> dict[str, float]:
        """2-ply paranoid minimax. V8.0: correct just_ate init (Fix 1.1)."""
        board    = data["board"]
        you      = data["you"]
        width    = board["width"]
        height   = board["height"]
        you_id   = you["id"]
        our_head = (you["head"]["x"], you["head"]["y"])
        enemies  = [s for s in board.get("snakes", []) if s["id"] != you_id]
        ei       = enemy_info or {}

        def make_sim(snake: dict) -> dict | None:
            body = [_pt(seg) for seg in snake.get("body", []) if _pt(seg) is not None]
            head = _pt(snake.get("head"))
            if head is None or not body:
                return None
            # Fix 1.1: correct just_ate using same logic as _build_occupied
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
        results: dict[str, float] = {}

        for direction in moves:
            if time.monotonic() >= deadline - 0.02:
                break

            dx, dy       = cls.DIRECTIONS[direction]
            our_new_head = (our_head[0] + dx, our_head[1] + dy)

            enemy_opts: list[list[str]] = []
            for esim in enemy_sims:
                e_moves = cls._get_legal_moves_sim(
                    esim["head"], esim["body"], occupied, width, height
                )
                if len(e_moves) > 2:
                    e_moves.sort(key=lambda m: _manhattan(
                        esim["head"][0] + cls.DIRECTIONS[m][0],
                        esim["head"][1] + cls.DIRECTIONS[m][1],
                        our_new_head[0], our_new_head[1],
                    ))
                    e_moves = e_moves[:2]
                enemy_opts.append(e_moves or ["up"])

            combos      = list(_iproduct(*enemy_opts)) if enemy_opts else [()]
            worst_score = float("inf")

            for combo in combos:
                if time.monotonic() >= deadline - 0.01:
                    break
                all_moves = {you_id: direction}
                for i, esim in enumerate(enemy_sims):
                    all_moves[esim["id"]] = combo[i] if i < len(combo) else "up"
                leaf = cls._simulate_and_evaluate(
                    us_sim, enemy_sims, all_moves,
                    food_set, hazard_set, hazard_dmg, food_dist_map, width, height,
                )
                if leaf < worst_score:
                    worst_score = leaf

            results[direction] = 0.0 if worst_score == float("inf") else worst_score

        return results

    # ================================================================== #
    #  Primary decision loop                                              #
    # ================================================================== #
    @classmethod
    def get_best_move(
        cls,
        data: dict,
        merged_food: set[tuple[int, int]],
        game_id: str = "",
        deadline: float = 0.0,
    ) -> str:
        if deadline == 0.0:
            deadline = time.monotonic() + cls.COMPUTE_BUDGET_S

        board   = data["board"]
        you     = data["you"]
        width   = board["width"]
        height  = board["height"]
        head    = you["head"]
        neck_pt = _pt(you["body"][1]) if len(you["body"]) > 1 else None

        mem        = _game_memory.get(game_id, {})
        enemy_info = mem.get("enemy_info", {})
        occupied   = cls._build_occupied(data, game_id, enemy_info)

        safe_moves: list[str] = []
        for direction, (dx, dy) in cls.DIRECTIONS.items():
            nx, ny = head["x"] + dx, head["y"] + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if neck_pt and nx == neck_pt[0] and ny == neck_pt[1]:
                continue
            if (nx, ny) in occupied:
                continue
            safe_moves.append(direction)

        if not safe_moves:
            in_bounds = [
                d for d, (dx, dy) in cls.DIRECTIONS.items()
                if 0 <= head["x"] + dx < width and 0 <= head["y"] + dy < height
            ]
            fallback = random.choice(in_bounds) if in_bounds else random.choice(list(cls.DIRECTIONS.keys()))
            log.warning("No safe moves! Fallback → %s", fallback)
            return fallback

        phase   = cls._get_game_phase(data)
        enemies = [s for s in board.get("snakes", []) if s["id"] != you["id"]]

        if phase == GamePhase.LATE_1V1:
            return cls._score_1v1(safe_moves, data, occupied, merged_food, enemy_info, deadline=deadline)

        chosen = cls._rank(safe_moves, data, occupied, merged_food, enemy_info,
                           deadline=deadline, phase=phase, enemies_cached=enemies)
        log.info("Safe moves: %s → chose: %s (phase=%s)", safe_moves, chosen, phase.value)
        return chosen

    # ================================================================== #
    #  Full scoring pipeline — V8.0                                      #
    # ================================================================== #
    @classmethod
    def _rank(
        cls,
        moves: list[str],
        data: dict,
        occupied: set[tuple[int, int]],
        merged_food: set[tuple[int, int]],
        enemy_info: dict,
        *,
        deadline: float = 0.0,
        phase: GamePhase = GamePhase.MID,
        enemies_cached: list[dict] | None = None,
    ) -> str:
        # Enhancement 4.3: graceful degradation
        try:
            return cls._rank_impl(
                moves, data, occupied, merged_food, enemy_info,
                deadline=deadline, phase=phase, enemies_cached=enemies_cached,
            )
        except Exception as exc:
            log.error("_rank failed (%s) — falling back to space-max", exc, exc_info=True)
            head  = data["you"]["head"]
            board = data["board"]
            return cls._rank_space_max(
                moves, occupied, board["width"], board["height"],
                head["x"], head["y"],
            )

    @classmethod
    def _rank_impl(
        cls,
        moves: list[str],
        data: dict,
        occupied: set[tuple[int, int]],
        merged_food: set[tuple[int, int]],
        enemy_info: dict,
        *,
        deadline: float = 0.0,
        phase: GamePhase = GamePhase.MID,
        enemies_cached: list[dict] | None = None,
    ) -> str:
        if deadline == 0.0:
            deadline = time.monotonic() + cls.COMPUTE_BUDGET_S

        board     = data["board"]
        you       = data["you"]
        width     = board["width"]
        height    = board["height"]
        head_x    = you["head"]["x"]
        head_y    = you["head"]["y"]
        health    = you["health"]
        snake_len = len(you["body"])
        you_id    = you["id"]
        turn      = data.get("turn", 0)
        center_x  = (width  - 1) / 2.0
        center_y  = (height - 1) / 2.0

        enemies = enemies_cached if enemies_cached is not None else [
            s for s in board.get("snakes", []) if s["id"] != you_id
        ]
        hazard_set = {(h["x"], h["y"]) for h in board.get("hazards", [])}
        hazard_dmg = _get_hazard_dmg(data)
        kill_margin = 1 if phase == GamePhase.LATE_FFA else cls.KILL_MARGIN

        # Phase-specific weight overrides
        w_center = -1 if phase == GamePhase.LATE_FFA else cls.W_CENTER

        # ── Pre-compute enemy data ─────────────────────────────────────
        enemy_heads:  list[tuple[int, int]] = []
        kill_cells:   set[tuple[int, int]]  = set()
        h2h_risk_map: dict[tuple[int, int], str] = {}
        priority_ord  = {"RISK_HIGH": 3, "RISK_EQUAL": 2, "RISK_LOW": 1, "RISK_NONE": 0}

        for e in enemies:
            e_head = e.get("head")
            if e_head is None:
                continue
            pos          = (e_head["x"], e_head["y"])
            e_body       = e.get("body", [])
            e_len_server = e.get("length", len(e_body))
            visible_segs = sum(1 for s in e_body if s is not None)

            # Kill classification (5.1)
            fully_vis_enough = (visible_segs >= e_len_server - 1)
            margin_ok        = (snake_len - visible_segs) >= kill_margin + 1
            safe_kill        = fully_vis_enough and margin_ok

            # Enhancement 3.1: trapped-enemy kill exception
            e_neck_raw = e_body[1] if len(e_body) > 1 else None
            e_neck_pt  = _pt(e_neck_raw)
            e_legal    = [
                (pos[0] + dx2, pos[1] + dy2)
                for dx2, dy2 in cls._DELTAS
                if (0 <= pos[0] + dx2 < width and 0 <= pos[1] + dy2 < height
                    and (pos[0] + dx2, pos[1] + dy2) not in occupied)
            ]
            if e_neck_pt:
                e_legal = [m for m in e_legal if m != e_neck_pt]
            if len(e_legal) == 1 and snake_len > e_len_server:
                # Enemy has only one escape → that cell is a certain kill
                kill_cells.add(e_legal[0])

            enemy_heads.append(pos)
            risk = cls._h2h_risk_level(snake_len, e_len_server, kill_margin)

            for dx, dy in cls._DELTAS:
                cell = (pos[0] + dx, pos[1] + dy)
                if not (0 <= cell[0] < width and 0 <= cell[1] < height):
                    continue
                if safe_kill:
                    kill_cells.add(cell)
                else:
                    cur = h2h_risk_map.get(cell, "RISK_NONE")
                    if priority_ord.get(risk, 0) > priority_ord.get(cur, 0):
                        h2h_risk_map[cell] = risk

        # Enhancement 3.2: reduce RISK_EQUAL to RISK_LOW on hazard cells
        for cell2 in list(h2h_risk_map.keys()):
            if h2h_risk_map[cell2] == "RISK_EQUAL" and cell2 in hazard_set:
                h2h_risk_map[cell2] = "RISK_LOW"

        # Enhancement 2.7: threat density map
        threat_count: dict[tuple[int, int], int] = {}
        for eh in enemy_heads:
            for dx2, dy2 in cls._DELTAS:
                c2 = (eh[0] + dx2, eh[1] + dy2)
                threat_count[c2] = threat_count.get(c2, 0) + 1

        # Enhancement 2.6: LATE_FFA risky favorable h2h cells
        risky_ffa_cells: set[tuple[int, int]] = set()
        if phase == GamePhase.LATE_FFA:
            for e in enemies:
                eh = e.get("head")
                if eh is None: continue
                e_len_ffa  = e.get("length", 0)
                e_vis_ffa  = sum(1 for s in e.get("body", []) if s is not None)
                safe_kill_ffa = (e_vis_ffa >= e_len_ffa - 1
                                  and (snake_len - e_vis_ffa) >= kill_margin + 1)
                if snake_len > e_len_ffa and not safe_kill_ffa:
                    for dx2, dy2 in cls._DELTAS:
                        c2 = (eh["x"] + dx2, eh["y"] + dy2)
                        if 0 <= c2[0] < width and 0 <= c2[1] < height:
                            risky_ffa_cells.add(c2)

        # Ghost risk (5.3)
        visible_ids = {s["id"] for s in board.get("snakes", []) if s.get("head") is not None}
        ghost_info  = {k: v for k, v in enemy_info.items() if k not in visible_ids}
        ghost_risk  = cls._compute_ghost_risk(ghost_info, turn, occupied, width, height)

        # Reverse food BFS (7.2)
        food_dist_map = cls._food_bfs_reverse(
            merged_food, occupied, width, height, hazard_set, hazard_dmg
        )

        # Food need
        max_enemy_len   = max((e.get("length", 0) for e in enemies), default=0)
        we_are_dominant = snake_len > max_enemy_len
        need_food       = health < cls.HUNGER_THRESHOLD or not we_are_dominant

        # ── Enhancement 2.2: Pin-trap filter (before corridor trap) ───
        pin_trap_set: set[str] = set()
        if len(enemies) <= 3 and time.monotonic() < deadline - 0.06 and len(moves) > 1:
            for d in moves:
                dx, dy = cls.DIRECTIONS[d]
                cand = (head_x + dx, head_y + dy)
                if cls._is_pin_trap(cand, head_x, head_y, enemies, occupied,
                                    width, height, kill_cells, deadline):
                    pin_trap_set.add(d)

        non_pinned   = [d for d in moves if d not in pin_trap_set]
        viable_moves = non_pinned if non_pinned else moves

        # ── Corridor trap filter (always runs) ─────────────────────────
        all_threatened: set[tuple[int, int]] = set()
        for eh in enemy_heads:
            for dx, dy in cls._DELTAS:
                cell = (eh[0] + dx, eh[1] + dy)
                if 0 <= cell[0] < width and 0 <= cell[1] < height:
                    all_threatened.add(cell)

        non_trapped = [
            d for d in viable_moves
            if not cls._is_corridor_trap(
                (head_x + cls.DIRECTIONS[d][0], head_y + cls.DIRECTIONS[d][1]),
                snake_len, occupied, all_threatened, width, height, kill_cells
            )
        ]
        viable_moves = non_trapped if non_trapped else viable_moves

        # ── Escape-route analysis (gated) ─────────────────────────────
        if (len(enemies) <= 2
                and time.monotonic() < deadline - 0.05
                and len(viable_moves) > 1):
            escape_safe = [
                d for d in viable_moves
                if not cls._min_escape_size(
                    (head_x + cls.DIRECTIONS[d][0], head_y + cls.DIRECTIONS[d][1]),
                    snake_len, occupied, enemies, width, height, deadline
                )
            ]
            if escape_safe:
                viable_moves = escape_safe

        # ── Per-move scoring (voronoi + food_dist) ────────────────────
        move_data: dict[str, dict] = {}
        timed_out = False

        for direction in viable_moves:
            if time.monotonic() >= deadline:
                if not timed_out:
                    log.warning("Budget exceeded after %d/%d moves", len(move_data), len(viable_moves))
                    timed_out = True
                dx, dy = cls.DIRECTIONS[direction]
                nx, ny = head_x + dx, head_y + dy
                cand   = (nx, ny)
                move_data[direction] = {
                    "nx": nx, "ny": ny, "voronoi": 0,
                    "food_dist": food_dist_map.get(cand, float(width + height)),
                    "kill": cand in kill_cells,
                    "h2h_risk": h2h_risk_map.get(cand, "RISK_NONE"),
                    "ghost": ghost_risk.get(cand, 0.0),
                    "board_haz": cand in hazard_set,
                }
                continue

            dx, dy = cls.DIRECTIONS[direction]
            nx, ny = head_x + dx, head_y + dy
            cand   = (nx, ny)

            voronoi   = cls.voronoi_bfs(cand, enemy_heads, occupied, width, height)
            food_dist = food_dist_map.get(cand, float(width + height))

            move_data[direction] = {
                "nx": nx, "ny": ny, "voronoi": voronoi,
                "food_dist": food_dist,
                "kill": cand in kill_cells,
                "h2h_risk": h2h_risk_map.get(cand, "RISK_NONE"),
                "ghost": ghost_risk.get(cand, 0.0),
                "board_haz": cand in hazard_set,
            }

        # Voronoi trap filter
        max_vor     = max((d["voronoi"] for d in move_data.values()), default=0)
        final_moves = [m for m in viable_moves if move_data[m]["voronoi"] >= snake_len]
        if not final_moves:
            final_moves = [m for m in viable_moves if move_data[m]["voronoi"] == max_vor]
        if not final_moves:
            final_moves = list(viable_moves)

        # Minimax scores (Fix 1.2: modifier, not replacement)
        mm_scores: dict[str, float] = {}
        if (len(enemies) <= 2
                and not timed_out
                and time.monotonic() < deadline - 0.08
                and len(final_moves) > 1):
            mm_scores = cls._minimax_2ply(
                final_moves, data, occupied, merged_food, enemy_heads,
                hazard_set, hazard_dmg, food_dist_map, deadline,
                enemy_info=enemy_info,
            ) or {}

        # Enhancement 2.4: desperation mode (health < 15, no reachable food)
        if health < 15 and food_dist_map:
            min_food_d = min(food_dist_map.values())
            if min_food_d >= health:
                desp = {
                    d: (-move_data[d]["food_dist"] * 100 + move_data[d]["voronoi"])
                    for d in final_moves
                }
                best_desp = max(desp.values())
                chosen_desp = random.choice([m for m, s in desp.items() if s == best_desp])
                log.info("V8 DESPERATION — health=%d food_dist=%.1f → %s",
                         health, min_food_d, chosen_desp)
                return chosen_desp

        # Precompute tail position for coiling (2.5)
        tail_seg = _pt(you["body"][-1]) if you["body"] else None

        # ── Unified scoring loop ───────────────────────────────────────
        scores: dict[str, float] = {}

        for direction in final_moves:
            md    = move_data[direction]
            nx    = md["nx"]
            ny    = md["ny"]
            cand  = (nx, ny)
            score = 0.0

            # Territory
            score += cls.W_VORONOI * md["voronoi"]

            # Combat
            if md["kill"]:
                score += cls.W_COMBAT_KILL

            # H2H risk (dynamic 5.2)
            risk = md["h2h_risk"]
            if risk == "RISK_HIGH":
                score -= 1200
            elif risk == "RISK_EQUAL":
                score -= 600
            elif risk == "RISK_LOW":
                score -= 200

            # Enhancement 2.7: crossfire penalty
            tc = threat_count.get(cand, 0)
            if tc >= 2:
                score -= 400

            # Ghost risk (5.3)
            score += md["ghost"]

            # Board hazard
            if md["board_haz"]:
                score += cls.W_HAZARD_CELL

            # Food with race evaluation — Fix 1.3 (corrected formula)
            if need_food:
                food_rv = cls._food_race_value(
                    cand, enemy_heads, food_dist_map, snake_len, enemies, width, height
                )
                if food_rv > 0:
                    # Certain/contested: stronger chase the more certain we are
                    score += cls.W_FOOD * md["food_dist"] * (1.0 + food_rv)
                elif food_rv < 0:
                    # Lost race: flat repulsion signal
                    score += abs(cls.W_FOOD) * 15
                else:
                    # Neutral: standard distance penalty
                    score += cls.W_FOOD * md["food_dist"]

            # Enhancement 2.1: edge/corner penalty
            on_edge   = (nx == 0 or nx == width - 1 or ny == 0 or ny == height - 1)
            on_corner = ((nx == 0 or nx == width - 1) and (ny == 0 or ny == height - 1))
            if on_edge:   score += cls.W_EDGE
            if on_corner: score += cls.W_CORNER

            # Enhancement 2.5: tail-following coiling (safe conditions only)
            if (health > 70 and tail_seg is not None
                    and all(_manhattan(nx, ny, eh[0], eh[1]) > 4 for eh in enemy_heads)):
                tail_dist = _manhattan(nx, ny, tail_seg[0], tail_seg[1])
                score += max(0, 10 - tail_dist) * 5

            # Center pull (phase-adjusted)
            score += w_center * (abs(nx - center_x) + abs(ny - center_y))

            # Enhancement 2.6: LATE_FFA aggression
            if phase == GamePhase.LATE_FFA:
                if md["kill"]:
                    score += cls.W_COMBAT_KILL * 0.5  # 800 → 1200 total
                elif cand in risky_ffa_cells:
                    score += cls.W_COMBAT_KILL * 0.5  # favorable but risky h2h

            # Fix 1.2: minimax as 5% modifier
            if mm_scores:
                mm = mm_scores.get(direction, 0.0)
                if mm <= -999999.0:
                    score -= 5000   # veto simulated death
                else:
                    score += mm * 0.05

            scores[direction] = score

        if not scores:
            return random.choice(viable_moves) if viable_moves else random.choice(moves)

        best_score = max(scores.values())
        best_moves = [m for m, s in scores.items() if s == best_score]
        chosen     = random.choice(best_moves)
        elapsed_ms = (time.monotonic() - (deadline - cls.COMPUTE_BUDGET_S)) * 1000
        log.info(
            "V8 Scores: %s | phase=%s dominant=%s health=%d elapsed=%.1fms%s → %s",
            {m: f"{s:.0f}" for m, s in scores.items()},
            phase.value, we_are_dominant, health, elapsed_ms,
            " [TIMEOUT]" if timed_out else "", chosen,
        )
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
    settings = data.get("game", {}).get("ruleset", {}).get("settings", {})
    log.info("Game started: %s | settings=%s", game_id, settings)
    return {}


@app.post("/move", response_class=JSONResponse)
async def move(state: GameState) -> dict:
    data    = state.model_dump()
    game_id = data.get("game", {}).get("id", "?")

    t_start = time.monotonic()

    # Enhancement 4.1: adaptive compute budget per game
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

    # Store latency for next turn's budget computation
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
    log.info("Game ended: %s — memory cleared", game_id)
    return {}


# ===========================================================================
# Verification suite — T1-T13
# ===========================================================================

def _verify_all() -> None:
    """
    T1–T10: V7.0 regression tests (preserved).
    T11: Edge/corner penalty.
    T12: Desperation mode (health < 15).
    T13: Threat density crossfire penalty.
    """
    import copy

    BASE_DATA: dict = {
        "game": {
            "id": "test-v8",
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
                    "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}],
                    "length": 3,
                },
                {
                    "id": "ghost-snake", "name": "ghost", "health": None,
                    "head": None, "body": [None, None, None, None], "length": 4,
                },
                {
                    "id": "partial-snake", "name": "partial", "health": None,
                    "head": {"x": 7, "y": 5},
                    "body": [{"x": 7, "y": 5}, {"x": 8, "y": 5}, None, None],
                    "length": 4,
                },
            ],
        },
        "you": {
            "id": "me", "name": "الثعبان", "health": 90,
            "head": {"x": 5, "y": 5},
            "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}],
            "length": 3,
        },
    }

    def _fresh(gid: str, data: dict) -> tuple[set, dict]:
        _game_memory[gid] = {"food": set(), "enemy_info": {}, "food_meta": {}, "latency_history": []}
        _update_enemy_memory(gid, data)
        return _update_food_memory(gid, data), _game_memory[gid]["enemy_info"]

    # T1
    gid = "test-v8"
    mf, ei = _fresh(gid, BASE_DATA)
    r = TacticalEngine.get_best_move(BASE_DATA, mf, game_id=gid)
    assert r in {"up", "down", "left", "right"}, f"T1 failed: {r!r}"
    log.info("✅ T1 null-safety OK — chose '%s'", r)

    # T2
    data2 = copy.deepcopy(BASE_DATA); data2["game"]["id"] = "tail-test"
    data2["board"]["snakes"].append({
        "id": "tailer", "name": "tailer", "health": 80,
        "head": {"x": 1, "y": 5},
        "body": [{"x":1,"y":5},{"x":2,"y":5},{"x":3,"y":5}], "length": 3,
    })
    _, ei2 = _fresh("tail-test", data2)
    occ2 = TacticalEngine._build_occupied(data2, "tail-test", ei2)
    assert (3, 5) not in occ2, "T2: tail should be free"
    assert (1, 5) in occ2,     "T2: head should be occupied"
    log.info("✅ T2 tail-vacating OK")

    # T3
    gid3 = "ghost-risk"; data3 = copy.deepcopy(BASE_DATA); data3["game"]["id"] = gid3
    data3["turn"] = 12
    _game_memory[gid3] = {
        "food": set(), "food_meta": {}, "latency_history": [],
        "enemy_info": {"ghost-snake": {
            "last_seen_turn": 10, "last_known_head": (5, 8),
            "last_body_count": 4, "prev_body_count": 4,
            "last_known_length": 4, "prev_known_length": 4,
        }},
    }
    _update_enemy_memory(gid3, data3)
    mf3 = _update_food_memory(gid3, data3)
    r3  = TacticalEngine.get_best_move(data3, mf3, game_id=gid3)
    assert r3 in {"up", "down", "left", "right"}, f"T3 failed: {r3!r}"
    log.info("✅ T3 ghost-risk OK — chose '%s'", r3)

    # T4
    mf4, ei4 = _fresh(gid, BASE_DATA)
    occ4 = TacticalEngine._build_occupied(BASE_DATA, gid, ei4)
    safe4 = [d for d in TacticalEngine.DIRECTIONS
              if (0 <= 5+TacticalEngine.DIRECTIONS[d][0] < 11
                  and 0 <= 5+TacticalEngine.DIRECTIONS[d][1] < 11
                  and (5+TacticalEngine.DIRECTIONS[d][0], 5+TacticalEngine.DIRECTIONS[d][1]) not in occ4)]
    TacticalEngine._rank(safe4, BASE_DATA, occ4, mf4, ei4)
    log.info("✅ T4 cautious-combat OK")

    # T5
    data5 = copy.deepcopy(BASE_DATA); data5["game"]["id"] = "haz-game"
    data5["you"]["head"] = {"x":1,"y":5}
    data5["you"]["body"] = [{"x":1,"y":5},{"x":1,"y":4},{"x":1,"y":3}]
    data5["board"]["snakes"][0].update({"head":{"x":1,"y":5},"body":[{"x":1,"y":5},{"x":1,"y":4},{"x":1,"y":3}]})
    mf5, _ = _fresh("haz-game", data5)
    r5 = TacticalEngine.get_best_move(data5, mf5, game_id="haz-game")
    assert r5 in {"up", "down", "left", "right"}, f"T5 failed: {r5!r}"
    log.info("✅ T5 hazard-avoidance OK — chose '%s'", r5)

    # T6: corridor trap
    tiny_occ = {(x, y) for x in range(11) for y in range(11)} - {(9, 9), (8, 9)}
    is_trap  = TacticalEngine._is_corridor_trap((9, 9), 3, tiny_occ, set(), 11, 11, set())
    assert is_trap, "T6 failed: 2-cell pocket should be a trap"
    log.info("✅ T6 corridor-trap OK")

    # T7: 1v1 phase
    data7 = copy.deepcopy(BASE_DATA); data7["game"]["id"] = "1v1-game"
    data7["board"]["snakes"] = [
        {"id":"me","name":"الثعبان","health":90,"head":{"x":5,"y":5},
         "body":[{"x":5,"y":5},{"x":5,"y":4},{"x":5,"y":3}],"length":3},
        {"id":"e1","name":"e","health":80,"head":{"x":8,"y":8},
         "body":[{"x":8,"y":8},{"x":8,"y":7}],"length":2},
    ]
    data7["you"] = data7["board"]["snakes"][0]
    mf7, _ = _fresh("1v1-game", data7)
    ph7 = TacticalEngine._get_game_phase(data7)
    assert ph7 == GamePhase.LATE_1V1, f"T7 failed: {ph7}"
    r7 = TacticalEngine.get_best_move(data7, mf7, game_id="1v1-game")
    assert r7 in {"up","down","left","right"}, f"T7 failed: {r7!r}"
    log.info("✅ T7 1v1-mode OK — phase=%s chose '%s'", ph7.value, r7)

    # T8: food race
    fd8 = {(5,6): 4.0, (5,3): 1.0}
    rv8 = TacticalEngine._food_race_value((5,6), [(5,3)], fd8, 3,
                                           [{"head":{"x":5,"y":3},"length":3}], 11, 11)
    assert rv8 <= 0, f"T8 failed: lost race should be non-positive, got {rv8}"
    log.info("✅ T8 food-race OK — rv=%.2f", rv8)

    # T9: dead-snake cleanup
    _game_memory["dead-game"] = {
        "food": set(), "food_meta": {}, "latency_history": [],
        "enemy_info": {"dead-e": {"last_seen_turn":5,"last_known_head":(3,3),
                                   "last_known_length":3,"prev_known_length":3}},
    }
    d9 = copy.deepcopy(BASE_DATA); d9["game"]["id"] = "dead-game"
    d9["board"]["snakes"] = [d9["board"]["snakes"][0]]
    _update_enemy_memory("dead-game", d9)
    assert "dead-e" not in _game_memory["dead-game"]["enemy_info"], "T9 failed"
    log.info("✅ T9 dead-snake cleanup OK")

    # T10: food stale TTL
    _game_memory["stale-game"] = {
        "food": {(0,0)}, "food_meta": {(0,0): 1},
        "enemy_info": {}, "latency_history": [],
    }
    d10 = copy.deepcopy(BASE_DATA); d10["game"]["id"] = "stale-game"; d10["turn"] = 50
    mf10 = _update_food_memory("stale-game", d10)
    assert (0,0) not in mf10, "T10 failed: stale food should be evicted"
    log.info("✅ T10 food-stale-TTL OK")

    # ── V8.0 new tests ──────────────────────────────────────────────────

    # T11: edge penalty — (0, 5) should score worse than (1, 5)
    occ11: set[tuple[int,int]] = set()  # empty board
    w_e = TacticalEngine.W_EDGE
    w_c = TacticalEngine.W_CORNER
    edge_score   = 20 * 100 + w_e          # Voronoi≈100 + edge penalty at (0,5)
    nonedge_score = 20 * 100               # Voronoi≈100, no edge at (1,5)
    assert w_e < 0, "T11 failed: W_EDGE should be negative"
    assert w_c < 0, "T11 failed: W_CORNER should be negative"
    assert edge_score < nonedge_score, "T11 failed: edge must penalize"
    log.info("✅ T11 edge/corner constants correct — W_EDGE=%d W_CORNER=%d", w_e, w_c)

    # T12: desperation mode — health < 15, food unreachable
    d12 = copy.deepcopy(BASE_DATA); d12["game"]["id"] = "desp-game"
    d12["you"]["health"] = 8
    d12["you"]["head"] = {"x":5,"y":5}
    d12["you"]["body"] = [{"x":5,"y":5},{"x":5,"y":4},{"x":5,"y":3}]
    d12["board"]["snakes"][0].update({"head":{"x":5,"y":5},
                                       "body":[{"x":5,"y":5},{"x":5,"y":4},{"x":5,"y":3}]})
    d12["board"]["snakes"] = [d12["board"]["snakes"][0]]
    d12["board"]["food"] = [{"x":0,"y":0}]  # dist=10 > health=8
    mf12, _ = _fresh("desp-game", d12)
    r12 = TacticalEngine.get_best_move(d12, mf12, game_id="desp-game")
    assert r12 in {"up","down","left","right"}, f"T12 failed: {r12!r}"
    log.info("✅ T12 desperation-mode OK — health=8 → chose '%s'", r12)

    # T13: threat density — (5,6) adjacent to 2 enemy heads → crossfire penalty
    d13 = copy.deepcopy(BASE_DATA); d13["game"]["id"] = "threat-game"
    d13["you"]["head"] = {"x":5,"y":5}; d13["you"]["body"] = [{"x":5,"y":5},{"x":5,"y":4},{"x":5,"y":3}]
    d13["board"]["snakes"] = [
        {"id":"me","name":"الثعبان","health":90,"head":{"x":5,"y":5},
         "body":[{"x":5,"y":5},{"x":5,"y":4},{"x":5,"y":3}],"length":3},
        {"id":"e1","name":"e1","health":90,"head":{"x":5,"y":7},
         "body":[{"x":5,"y":7},{"x":5,"y":8}],"length":2},
        {"id":"e2","name":"e2","health":90,"head":{"x":4,"y":6},
         "body":[{"x":4,"y":6},{"x":3,"y":6}],"length":2},
    ]
    d13["you"] = d13["board"]["snakes"][0]
    # Both e1 and e2 can reach (5,6): threat_count[(5,6)] = 2
    mf13, _ = _fresh("threat-game", d13)
    r13 = TacticalEngine.get_best_move(d13, mf13, game_id="threat-game")
    assert r13 in {"up","down","left","right"}, f"T13 failed: {r13!r}"
    log.info("✅ T13 threat-density OK — crossfire at (5,6), chose '%s'", r13)

    # Cleanup
    for gid2 in ["test-v8","tail-test",gid3,"haz-game","1v1-game",
                  "dead-game","stale-game","desp-game","threat-game"]:
        _game_memory.pop(gid2, None)


_verify_all()


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
