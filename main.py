"""
Battlesnake Blackout 2026 — High-Performance FastAPI Server
Survival Logic V10.0 | APEX PREDATOR Protocol

3-Layer Architecture:
  1. SURVIVAL FILTER (_is_safe): Strict non-negotiable rules + N-Turn Lookahead.
  2. STRATEGY SCORER (_score_moves): Single weighted float score based on heuristics.
  3. PHASE DETECTOR: Dynamic weights based on game state.
"""

import heapq
import logging
import random
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

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
app = FastAPI(title="الثعبان — Battlesnake Blackout 2026", version="10.0.0")

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
    "version":    "10.0.0",
}

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
FOOD_STALE_TTL = 40
COMPUTE_BUDGET_S = 0.230  # Leave 20ms safety margin (250ms total)

# ---------------------------------------------------------------------------
# Per-game memory
# ---------------------------------------------------------------------------
_game_memory: dict[str, dict] = {}


# ===========================================================================
# Enums & Dataclasses
# ===========================================================================
class GamePhase(Enum):
    EARLY    = "early"
    MID      = "mid"
    LATE_1V1 = "late_1v1"
    LATE_FFA = "late_ffa"


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
    occupied:             set              # solid blocked cells (OOB + bodies)
    hazard_set:           set
    hazard_dmg:           int
    enemy_heads:          list             # list[tuple[int,int]]
    enemy_data:           list             # list[dict]
    visible_food:         set
    merged_food:          set
    phase:                GamePhase
    weights:              dict
    deadline:             float
    view_radius:          int
    enemy_info:           dict             # raw memory entry
    ghost_zones:          set              # cells near last known enemy positions
    unseen_cells:         set              # cells outside view radius


# ===========================================================================
# Phase Weights
# ===========================================================================
PHASE_WEIGHTS = {
    GamePhase.EARLY: {
        "W_VORONOI": 15.0, "W_FOOD": 40.0, "W_KILL": 100.0, "W_TAIL": 5.0,
        "W_CENTER": 15.0, "W_EDGE": 15.0, "W_CORNER": 40.0, "W_HAZARD": 1000.0,
        "W_GHOST": 50.0, "W_CONSTRICT": 0.0, "W_PIN": 0.0, "W_FOG_RISK": 20.0
    },
    GamePhase.MID: {
        "W_VORONOI": 25.0, "W_FOOD": 25.0, "W_KILL": 250.0, "W_TAIL": 15.0,
        "W_CENTER": 5.0, "W_EDGE": 15.0, "W_CORNER": 40.0, "W_HAZARD": 1000.0,
        "W_GHOST": 50.0, "W_CONSTRICT": 15.0, "W_PIN": 200.0, "W_FOG_RISK": 35.0
    },
    GamePhase.LATE_1V1: {
        "W_VORONOI": 15.0, "W_FOOD": 10.0, "W_KILL": 1500.0, "W_TAIL": 40.0,
        "W_CENTER": 0.0, "W_EDGE": 10.0, "W_CORNER": 25.0, "W_HAZARD": 1000.0,
        "W_GHOST": 70.0, "W_CONSTRICT": 55.0, "W_PIN": 1500.0, "W_FOG_RISK": 50.0
    },
    GamePhase.LATE_FFA: {
        "W_VORONOI": 30.0, "W_FOOD": 20.0, "W_KILL": 400.0, "W_TAIL": 20.0,
        "W_CENTER": 0.0, "W_EDGE": 15.0, "W_CORNER": 40.0, "W_HAZARD": 1000.0,
        "W_GHOST": 50.0, "W_CONSTRICT": 0.0, "W_PIN": 0.0, "W_FOG_RISK": 30.0
    }
}


# ===========================================================================
# Utility helpers
# ===========================================================================
def _pt(seg: Any) -> tuple[int, int] | None:
    if seg is None: return None
    return (seg["x"], seg["y"])

def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)

def _get_view_radius(data: dict) -> int:
    try: return int(data["game"]["ruleset"]["settings"]["viewRadius"])
    except (KeyError, TypeError, ValueError): return 5

def _get_hazard_dmg(data: dict) -> int:
    try: return int(data["game"]["ruleset"]["settings"]["hazardDamagePerTurn"])
    except (KeyError, TypeError, ValueError): return 0

def _is_in_view(px: int, py: int, hx: int, hy: int, radius: int) -> bool:
    return _manhattan(px, py, hx, hy) <= radius


# ===========================================================================
# Memory updates
# ===========================================================================
def _update_food_memory(game_id: str, data: dict) -> set[tuple[int, int]]:
    mem = _game_memory.setdefault(
        game_id, {"food": set(), "enemy_info": {}, "food_meta": {}, "fallback_history": []}
    )
    prev_food: set = mem.setdefault("food", set())
    food_meta: dict = mem.setdefault("food_meta", {})

    you    = data["you"]
    head   = you.get("head") or {}
    hx: int = head.get("x", 0)
    hy: int = head.get("y", 0)
    radius = _get_view_radius(data)
    turn   = data.get("turn", 0)

    visible_food = {(f["x"], f["y"]) for f in data.get("board", {}).get("food", [])}

    new_memory = set()
    new_meta = {}

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
        if pos not in new_memory:
            new_memory.add(pos)
            new_meta[pos] = turn

    mem["food"] = new_memory
    mem["food_meta"] = new_meta
    return new_memory


def _update_enemy_memory(game_id: str, data: dict) -> dict:
    mem = _game_memory.setdefault(
        game_id, {"food": set(), "enemy_info": {}, "food_meta": {}, "fallback_history": []}
    )
    enemy_info: dict = mem.setdefault("enemy_info", {})
    
    you_id = data["you"]["id"]
    snakes = data.get("board", {}).get("snakes", [])
    turn   = data.get("turn", 0)

    live_ids = set()
    for s in snakes:
        sid = s["id"]
        if sid == you_id:
            continue
        live_ids.add(sid)
        head = _pt(s.get("head"))
        if head:
            enemy_info[sid] = {
                "last_head": head,
                "last_seen_turn": turn,
                "length": s.get("length", 0)
            }

    # Purge dead snakes
    dead = set(enemy_info.keys()) - live_ids
    for sid in dead:
        del enemy_info[sid]

    return enemy_info


# ===========================================================================
# Engine Core
# ===========================================================================
class TacticalEngine:
    COMPUTE_BUDGET_S = 0.230
    DIRECTIONS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

    @classmethod
    def get_best_move(cls, data: dict, merged_food: set, game_id: str, deadline: float) -> str:
        ctx = cls._build_context(data, merged_food, game_id, deadline)

        if ctx.turn <= 5:
            move = cls._get_opening_move(ctx)
            if move: return move

        safe_moves = []
        for d in cls.DIRECTIONS.keys():
            if cls._is_safe(d, ctx):
                safe_moves.append(d)

        # Bug 4 Fix: Fallback forensics & intelligent fallback
        if not safe_moves:
            log.warning("No completely safe moves found! Executing intelligent fallback.")
            return cls._intelligent_fallback(ctx, game_id)

        scored_moves = []
        for move in safe_moves:
            score = cls._score_move(move, ctx)
            scored_moves.append((score, move))
        
        scored_moves.sort(reverse=True)
        best_score, best_move = scored_moves[0]
        
        # Log to fallback history
        history = _game_memory[game_id].setdefault("fallback_history", [])
        history.append(f"T{ctx.turn}: {len(safe_moves)} safe, chose {best_move} (score {best_score:.1f}, {ctx.phase.value})")
        if len(history) > 10:
            history.pop(0)

        log.info(f"Safe moves: {safe_moves} → chose: {best_move} (phase={ctx.phase.value}, score={best_score:.1f})")
        return best_move

    @classmethod
    def _build_context(cls, data: dict, merged_food: set, game_id: str, deadline: float) -> GameContext:
        board = data.get("board", {})
        you = data["you"]
        w = board.get("width", 11)
        h = board.get("height", 11)
        turn = data.get("turn", 0)

        our_body = [_pt(s) for s in you.get("body", [])]
        our_head = our_body[0] if our_body else (0, 0)
        our_len = you.get("length", 1)
        our_health = you.get("health", 100)
        
        our_tail = our_body[-1] if len(our_body) > 1 and our_body[-1] else None

        occupied = set()
        for b in our_body[:-1]:
            if b: occupied.add(b)
        
        # Tail vacating logic (only solid if just ate)
        if our_tail:
            # We know exactly if we ate (health is 100)
            if our_health == 100:
                occupied.add(our_tail)

        enemy_heads = []
        enemy_data = []
        
        snakes = board.get("snakes", [])
        num_snakes = len(snakes)

        for s in snakes:
            if s["id"] == you["id"]: continue
            e_len = s.get("length", 1)
            e_body = [_pt(seg) for seg in s.get("body", [])]
            e_head = e_body[0] if e_body else None
            
            if e_head:
                enemy_heads.append(e_head)
                for b in e_body[:-1]:
                    if b: occupied.add(b)
                # Opponent tail logic (default vacate unless we saw them grow - simplifying to always vacate unless 100)
                e_health = s.get("health", 100)
                e_tail = e_body[-1] if len(e_body) > 1 and e_body[-1] else None
                if e_tail and e_health == 100:
                    occupied.add(e_tail)
                    
                enemy_data.append({
                    "id": s["id"],
                    "head_pos": e_head,
                    "length": e_len,
                    "health": e_health,
                    "body": e_body
                })

        hazard_set = {(f["x"], f["y"]) for f in board.get("hazards", [])}
        hazard_dmg = _get_hazard_dmg(data)

        # OOB bounds to occupied
        for x in range(w):
            occupied.add((x, -1))
            occupied.add((x, h))
        for y in range(h):
            occupied.add((-1, y))
            occupied.add((w, y))

        view_radius = _get_view_radius(data)
        
        # Ghost zones (unseen enemies)
        enemy_info = _game_memory.get(game_id, {}).get("enemy_info", {})
        ghost_zones = set()
        for eid, einfo in enemy_info.items():
            last_seen = einfo.get("last_seen_turn", 0)
            if turn - last_seen > 0:  # currently unseen
                lh = einfo.get("last_head")
                if lh:
                    # Mark a radius 2 around last known head as ghost zone
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            if abs(dx) + abs(dy) <= 2:
                                gx, gy = lh[0]+dx, lh[1]+dy
                                if 0 <= gx < w and 0 <= gy < h:
                                    ghost_zones.add((gx, gy))
                                    
        unseen_cells = set()
        for x in range(w):
            for y in range(h):
                if not _is_in_view(x, y, our_head[0], our_head[1], view_radius):
                    unseen_cells.add((x, y))

        # Phase Detection
        if turn < 20: phase = GamePhase.EARLY
        elif num_snakes == 2 and turn >= 20: phase = GamePhase.LATE_1V1
        elif num_snakes > 2 and turn > 60: phase = GamePhase.LATE_FFA
        else: phase = GamePhase.MID

        return GameContext(
            our_head=our_head, our_body=our_body, our_len=our_len,
            our_health=our_health, our_tail=our_tail,
            width=w, height=h, turn=turn, occupied=occupied,
            hazard_set=hazard_set, hazard_dmg=hazard_dmg,
            enemy_heads=enemy_heads, enemy_data=enemy_data,
            visible_food={(f["x"], f["y"]) for f in board.get("food", [])},
            merged_food=merged_food, phase=phase, weights=PHASE_WEIGHTS[phase],
            deadline=deadline, view_radius=view_radius, enemy_info=enemy_info,
            ghost_zones=ghost_zones, unseen_cells=unseen_cells
        )

    # -----------------------------------------------------------------------
    # Layer 1: Survival Filter
    # -----------------------------------------------------------------------
    @classmethod
    def _is_safe(cls, direction: str, ctx: GameContext) -> bool:
        dx, dy = cls.DIRECTIONS[direction]
        nx, ny = ctx.our_head[0] + dx, ctx.our_head[1] + dy
        cand = (nx, ny)

        # 1. Bounds & Solid Body Checks
        if cand in ctx.occupied:
            return False

        # 2. Hazard Check
        if cand in ctx.hazard_set:
            if ctx.our_health <= ctx.hazard_dmg + 5:
                return False
            # Only enter hazard if health > 30, otherwise try to avoid (soft handled in scorer usually, but strict here if health low)
            if ctx.our_health < 30:
                # We veto hazard entirely if health < 30 unless it's fallback.
                return False

        # 3. Head-to-Head Check
        for e in ctx.enemy_data:
            eh = e["head_pos"]
            if eh and _manhattan(nx, ny, eh[0], eh[1]) == 1:
                if e["length"] >= ctx.our_len:
                    return False

        # 4. Corridor Trap Check (Simple)
        if cls._is_corridor_trap(cand, ctx):
            return False

        # 5. [A] Deep Escape Route Verification (N-Turn Lookahead)
        if not cls._deep_escape_check(cand, ctx):
            return False

        return True

    @classmethod
    def _is_corridor_trap(cls, cand: tuple[int, int], ctx: GameContext) -> bool:
        """Fast check if the immediate cell leads to an inescapable dead end right now."""
        # Flood fill current occupied
        space = cls._flood_fill(cand, ctx.occupied, limit=ctx.our_len + 3, deadline=ctx.deadline)
        if space < ctx.our_len + 3:
            return True
        return False

    @classmethod
    def _deep_escape_check(cls, cand: tuple[int, int], ctx: GameContext, depth: int = 3) -> bool:
        """
        N-Turn Lookahead: Simulates the board after 1..N turns by relaxing tails.
        Ensures we have >= our_len + 2 space at every future step.
        """
        for n in range(1, depth + 1):
            if time.monotonic() > ctx.deadline - 0.01:
                return False # Bug 2 Fix: Assume danger on timeout

            sim_occupied = set()
            # Add bounds
            for x in range(ctx.width):
                sim_occupied.add((x, -1))
                sim_occupied.add((x, ctx.height))
            for y in range(ctx.height):
                sim_occupied.add((-1, y))
                sim_occupied.add((ctx.width, y))
            
            # Simulate our body (tail moves forward by n)
            body_len = len(ctx.our_body)
            keep_len = max(1, body_len - n)
            for i in range(keep_len):
                if ctx.our_body[i]: sim_occupied.add(ctx.our_body[i])
            # Note: Do not add cand to sim_occupied, as we flood fill FROM cand

            
            # Simulate enemy bodies
            for e in ctx.enemy_data:
                e_body = e["body"]
                e_blen = len(e_body)
                e_keep = max(1, e_blen - n)
                for i in range(e_keep):
                    if e_body[i]: sim_occupied.add(e_body[i])
                    
            space = cls._flood_fill(cand, sim_occupied, limit=ctx.our_len + 2, deadline=ctx.deadline)
            if space < ctx.our_len + 2:
                return False
                
        return True

    @classmethod
    def _flood_fill(cls, start: tuple[int, int], occupied: set, limit: int, deadline: float) -> int:
        if start in occupied: return 0
        visited = {start}
        queue = deque([start])
        count = 0
        
        while queue:
            if count >= limit: break
            if time.monotonic() > deadline - 0.005:
                break # Timeout -> return what we have (might fail threshold -> conservative)
                
            curr = queue.popleft()
            count += 1
            
            for dx, dy in ((0,1), (0,-1), (-1,0), (1,0)):
                nx, ny = curr[0]+dx, curr[1]+dy
                cand = (nx, ny)
                if cand not in occupied and cand not in visited:
                    visited.add(cand)
                    queue.append(cand)
        return count


    # -----------------------------------------------------------------------
    # Layer 2: Strategy Scorer
    # -----------------------------------------------------------------------
    @classmethod
    def _score_move(cls, move: str, ctx: GameContext) -> float:
        dx, dy = cls.DIRECTIONS[move]
        cand = (ctx.our_head[0]+dx, ctx.our_head[1]+dy)
        w = ctx.weights
        
        score = 0.0
        
        # 1. Voronoi Area
        v_area = cls._flood_fill(cand, ctx.occupied, limit=ctx.width * ctx.height, deadline=ctx.deadline)
        score += v_area * w["W_VORONOI"]
        
        # 2. Food Proximity
        # Fast hazard-agnostic BFS for nearest food
        food_dist = cls._bfs_dist(cand, ctx.merged_food, ctx.occupied, limit=50)
        if food_dist < 999:
            score -= food_dist * w["W_FOOD"]
            # [F] Food Race
            score += cls._food_race_score(cand, food_dist, ctx)
        
        # Absolute hunger priority
        if ctx.our_health < 30 and food_dist < 999:
            score -= food_dist * w["W_FOOD"] * 5 # Override
            
        # 3. Kill Opportunity
        for e in ctx.enemy_data:
            eh = e["head_pos"]
            if eh and e["length"] < ctx.our_len and _manhattan(cand[0], cand[1], eh[0], eh[1]) == 1:
                score += w["W_KILL"]
                
        # 4. Tail Following [C] Coiling Defense
        score += cls._coiling_score(cand, ctx)
        
        # 5. Positioning Penalties
        cx, cy = ctx.width // 2, ctx.height // 2
        dist_to_center = _manhattan(cand[0], cand[1], cx, cy)
        score -= dist_to_center * w["W_CENTER"]
        
        is_edge = (cand[0] == 0 or cand[0] == ctx.width - 1 or cand[1] == 0 or cand[1] == ctx.height - 1)
        is_corner = (cand[0] in (0, ctx.width-1)) and (cand[1] in (0, ctx.height-1))
        
        if is_edge: score -= w["W_EDGE"]
        if is_corner: score -= w["W_CORNER"]
        if cand in ctx.hazard_set: score -= w["W_HAZARD"]
        if cand in ctx.ghost_zones: score -= w["W_GHOST"]
        
        # 6. 1V1 Tactics
        if ctx.phase == GamePhase.LATE_1V1 and len(ctx.enemy_data) == 1:
            score += cls._constriction_score(cand, ctx)
            score += cls._pin_bonus(cand, ctx)
            score += cls._executioner_score(cand, ctx)
            
        # 7. [B] Fog of War Conservative Bias
        score -= cls._fog_risk_score(cand, ctx)
        
        return score

    @classmethod
    def _bfs_dist(cls, start: tuple[int, int], targets: set, occupied: set, limit: int = 100) -> int:
        if not targets: return 999
        if start in targets: return 0
        
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            curr, dist = queue.popleft()
            if dist > limit: break
            
            for dx, dy in ((0,1), (0,-1), (-1,0), (1,0)):
                nx, ny = curr[0]+dx, curr[1]+dy
                cand = (nx, ny)
                if cand in targets:
                    return dist + 1
                if cand not in occupied and cand not in visited:
                    visited.add(cand)
                    queue.append((cand, dist + 1))
        return 999

    @classmethod
    def _food_race_score(cls, cand: tuple[int, int], our_dist: int, ctx: GameContext) -> float:
        if our_dist == 999 or not ctx.enemy_data: return 0.0
        
        # Find which food we are pathing to (approximate by just taking nearest in merged)
        best_food = None
        best_fd = 999
        for f in ctx.merged_food:
            fd = _manhattan(cand[0], cand[1], f[0], f[1])
            if fd < best_fd:
                best_fd, best_food = fd, f
                
        if not best_food: return 0.0
        
        closest_enemy_dist = 999
        longest_enemy_len = 0
        for e in ctx.enemy_data:
            eh = e["head_pos"]
            if eh:
                edist = _manhattan(eh[0], eh[1], best_food[0], best_food[1])
                if edist < closest_enemy_dist:
                    closest_enemy_dist = edist
                    longest_enemy_len = e["length"]
                    
        # If we are strictly closer
        if our_dist < closest_enemy_dist:
            return ctx.weights["W_FOOD"]  # Effectively doubles food weight
            
        # If enemy is closer or tied, and is longer/equal -> bail
        if our_dist >= closest_enemy_dist and longest_enemy_len >= ctx.our_len:
            return -ctx.weights["W_FOOD"] * 10 # Massive penalty, don't race
            
        return 0.0

    @classmethod
    def _coiling_score(cls, cand: tuple[int, int], ctx: GameContext) -> float:
        if not ctx.our_tail: return 0.0
        
        longer_enemy_exists = any(e["length"] > ctx.our_len for e in ctx.enemy_data)
        highly_defensive = (ctx.our_health > 40)
        
        if highly_defensive or longer_enemy_exists:
            # Don't coil if starving and food is right there
            if ctx.our_health < 25: return 0.0
            if ctx.our_health < 50 and any(_manhattan(cand[0], cand[1], f[0], f[1]) <= 2 for f in ctx.visible_food):
                return 0.0
                
            dist_to_tail = cls._bfs_dist(cand, {ctx.our_tail}, ctx.occupied, limit=30)
            if dist_to_tail < 999:
                return ctx.weights["W_TAIL"] * (30 - dist_to_tail)
        return 0.0

    @classmethod
    def _constriction_score(cls, cand: tuple[int, int], ctx: GameContext) -> float:
        # Simplistic voronoi diff
        e = ctx.enemy_data[0]
        eh = e["head_pos"]
        if not eh: return 0.0
        
        # If we move closer to enemy, we constrict them. Simple manhattan proxy for performance.
        dist_now = _manhattan(ctx.our_head[0], ctx.our_head[1], eh[0], eh[1])
        dist_next = _manhattan(cand[0], cand[1], eh[0], eh[1])
        
        if dist_next < dist_now:
            return ctx.weights["W_CONSTRICT"]
        return 0.0

    @classmethod
    def _pin_bonus(cls, cand: tuple[int, int], ctx: GameContext) -> float:
        e = ctx.enemy_data[0]
        eh = e["head_pos"]
        if not eh or e["length"] >= ctx.our_len: return 0.0
        
        # Does enemy have 1 exit?
        e_escapes = 0
        for dx, dy in ((0,1), (0,-1), (-1,0), (1,0)):
            nx, ny = eh[0]+dx, eh[1]+dy
            if (nx, ny) not in ctx.occupied:
                e_escapes += 1
                
        if e_escapes == 1 and _manhattan(cand[0], cand[1], eh[0], eh[1]) <= 2:
            return ctx.weights["W_PIN"]
        return 0.0

    @classmethod
    def _executioner_score(cls, cand: tuple[int, int], ctx: GameContext) -> float:
        e = ctx.enemy_data[0]
        if e["length"] >= ctx.our_len or e["health"] >= 55: return 0.0
        eh = e["head_pos"]
        if not eh: return 0.0
        
        # If we step on the cell that is their shortest path to food
        food_dist = cls._bfs_dist(eh, ctx.merged_food, ctx.occupied, limit=20)
        if food_dist < 999:
            cand_to_food = cls._bfs_dist(cand, ctx.merged_food, ctx.occupied, limit=20)
            if cand_to_food < food_dist and _manhattan(cand[0], cand[1], eh[0], eh[1]) == 1:
                return ctx.weights["W_KILL"] * 2 # Massive bonus
        return 0.0

    @classmethod
    def _fog_risk_score(cls, cand: tuple[int, int], ctx: GameContext) -> float:
        # Bug 1 Fix: Correct operator precedence in complex checks (already simplified logic here)
        # If the candidate relies on unseen cells, apply penalty. Note it returns penalty to SUBTRACT.
        if cand in ctx.unseen_cells:
            return ctx.weights["W_FOG_RISK"]
            
        # Check adjacent cells for fog
        for dx, dy in ((0,1), (0,-1), (-1,0), (1,0)):
            adj = (cand[0]+dx, cand[1]+dy)
            if adj in ctx.unseen_cells and adj not in ctx.occupied:
                return ctx.weights["W_FOG_RISK"] * 0.5
        
        return 0.0

    # -----------------------------------------------------------------------
    # Opening & Fallback
    # -----------------------------------------------------------------------
    @classmethod
    def _get_opening_move(cls, ctx: GameContext) -> str | None:
        """Turn 1-5: Rush closest safe visible food, else center."""
        best_dir = None
        best_dist = 999
        
        for d, (dx, dy) in cls.DIRECTIONS.items():
            cand = (ctx.our_head[0]+dx, ctx.our_head[1]+dy)
            if not cls._is_safe(d, ctx): continue
            
            if ctx.visible_food:
                fd = cls._bfs_dist(cand, ctx.visible_food, ctx.occupied, limit=20)
                if fd < best_dist:
                    best_dist = fd
                    best_dir = d
            else:
                cx, cy = ctx.width // 2, ctx.height // 2
                cd = _manhattan(cand[0], cand[1], cx, cy)
                if cd < best_dist:
                    best_dist = cd
                    best_dir = d
                    
        return best_dir

    @classmethod
    def _intelligent_fallback(cls, ctx: GameContext, game_id: str) -> str:
        """Bug 4 Fix: When zero safe moves, choose the least damaging one."""
        # 1. Avoid walls/static bodies at all costs
        # 2. Avoid H2H with larger enemies
        # 3. Hazards are better than walls
        
        best_move = random.choice(list(cls.DIRECTIONS.keys())) # default
        best_score = -9999
        
        for d, (dx, dy) in cls.DIRECTIONS.items():
            nx, ny = ctx.our_head[0]+dx, ctx.our_head[1]+dy
            cand = (nx, ny)
            
            # Absolute death (walls)
            if nx < 0 or ny < 0 or nx >= ctx.width or ny >= ctx.height:
                continue
                
            score = 0
            
            if cand in ctx.occupied: score -= 1000
            if cand in ctx.hazard_set: score -= 100
            
            h2h = False
            for e in ctx.enemy_data:
                eh = e["head_pos"]
                if eh and _manhattan(nx, ny, eh[0], eh[1]) == 1:
                    if e["length"] >= ctx.our_len: score -= 500
                    h2h = True
                    
            if score > best_score:
                best_score = score
                best_move = d
                
        # Log Fallback forensics [G]
        history = _game_memory.get(game_id, {}).get("fallback_history", [])
        log.error(f"FALLBACK FORENSICS triggered at Turn {ctx.turn}. History: {history}")
        
        return best_move


# ===========================================================================
# FastAPI Endpoints
# ===========================================================================
@app.get("/")
def on_info():
    return JSONResponse(SNAKE_INFO)


@app.post("/start")
def on_start(state: GameState):
    data = state.model_dump()
    game_id = data.get("game", {}).get("id", "unknown")
    _game_memory[game_id] = {"food": set(), "enemy_info": {}, "food_meta": {}, "fallback_history": []}
    return "ok"


@app.post("/move")
def on_move(state: GameState):
    t0 = time.monotonic()
    deadline = t0 + COMPUTE_BUDGET_S

    data = state.model_dump()
    game_id = data.get("game", {}).get("id", "unknown")

    _update_enemy_memory(game_id, data)
    merged_food = _update_food_memory(game_id, data)

    try:
        direction = TacticalEngine.get_best_move(data, merged_food, game_id, deadline)
    except Exception as e:
        log.exception(f"Exception in get_best_move: {e}")
        direction = "up"

    elapsed_ms = (time.monotonic() - t0) * 1000
    log.info(f"Turn {data.get('turn')} | {direction} | {elapsed_ms:.1f} ms")

    return JSONResponse({"move": direction})


@app.post("/end")
def on_end(state: GameState):
    data = state.model_dump()
    game_id = data.get("game", {}).get("id", "unknown")
    _game_memory.pop(game_id, None)
    return "ok"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
