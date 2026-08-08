"""
Battlesnake Blackout 2026 — Tactical Engine
=============================================================================
3-Layer architecture:
  1. SURVIVAL FILTER  (_is_safe / _is_certain_death)  — non-negotiable rules
     plus a graduated fallback so we never discard useful space analysis.
  2. STRATEGY SCORER  (_score_move)                   — single weighted score.
  3. PHASE DETECTOR   (GamePhase / PHASE_WEIGHTS)      — dynamic weights.

This file replaces the previous rewrite, which technically implemented the
3-layer shape but had one serious behavioural bug: Layer 1's corridor/escape
checks were a hard veto with no middle ground. Once the board got crowded
(mid/late game on an 11x11 board, especially with 2-3 other snakes), it was
common for *all four* directions to fail that strict check on the same turn.
When that happened the engine threw away all of its space/food/kill analysis
and picked a move almost blindly. In local batch testing that "no strict-safe
move" branch fired dozens of times across a handful of games — i.e. most of
the engine's intelligence was being bypassed constantly, not just in genuine
corner cases.

Fix: Layer 1 now has two tiers.
  Tier A (certain death — OOB / body collision / lethal hazard / losing H2H):
    never entered, no matter what.
  Tier B (corridor trap / N-turn escape margin / risky-but-survivable hazard):
    preferred against, but if it turns out *every* direction is "tight" this
    turn, we still hand the full strategy scorer the Tier-A-safe set instead
    of falling back to a blind heuristic. The scorer's own Voronoi term will
    naturally favour whichever tight option has the most room.
  Only if literally every direction fails Tier A (a genuine, inescapable box)
  do we drop to `_last_resort_move`, which is itself space- and risk-aware
  rather than a plain "avoid occupied cells" fallback.
=============================================================================
"""

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
app = FastAPI(title="الثعبان — Battlesnake Blackout 2026", version="11.1.0")


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
    "version":    "11.1.0",
}

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
FOOD_STALE_TTL      = 40     # turns a remembered-but-unseen food stays "believed"
COMPUTE_BUDGET_S    = 0.230  # leave ~20ms safety margin out of a 250ms turn

HUNGER_CRITICAL      = 30    # [F] below this, food distance overrides scoring
HAZARD_SOFT_HEALTH   = 30    # Layer1: never *choose* hazard below this health
CORRIDOR_MARGIN      = 3     # Layer1: veto corridor if space < our_len + this
ESCAPE_MARGIN        = 2     # [A]/N-turn lookahead: required space margin
ESCAPE_DEPTH         = 3     # [A]: look 3 turns ahead

COIL_DISABLE_HEALTH  = 25    # [C]: below this, never coil — go get food
COIL_FOOD_HEALTH     = 50    # [C]: don't coil past nearby food below this health
COIL_FOOD_DIST       = 2     # [C]: "nearby" food distance threshold

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
    """Built once per turn in _build_context and passed everywhere else.
    Only fields actually read downstream are kept (Bug 6 cleanup)."""
    our_head:    tuple[int, int]
    our_body:    list
    our_len:     int
    our_health:  int
    our_tail:    tuple[int, int] | None
    width:       int
    height:      int
    turn:        int
    occupied:    set              # solid blocked cells (OOB ring + bodies)
    hazard_set:  set
    hazard_dmg:  int
    enemy_data:  list              # list[dict]: id, head_pos, length, health, body
    visible_food: set
    merged_food: set
    phase:       GamePhase
    weights:     dict
    deadline:    float
    ghost_zones: set               # cells near a currently-unseen enemy's last head
    unseen_cells: set              # cells outside our view radius


# ===========================================================================
# Phase Weights (tunable — see PHASE_WEIGHTS self-play tuning note at bottom)
# ===========================================================================
PHASE_WEIGHTS = {
    GamePhase.EARLY: {
        "W_VORONOI": 15.0, "W_FOOD": 40.0, "W_KILL": 100.0, "W_TAIL": 5.0,
        "W_CENTER": 15.0, "W_EDGE": 15.0, "W_CORNER": 40.0, "W_HAZARD": 1000.0,
        "W_GHOST": 50.0, "W_CONSTRICT": 0.0, "W_PIN": 0.0, "W_FOG_RISK": 20.0,
    },
    GamePhase.MID: {
        "W_VORONOI": 25.0, "W_FOOD": 25.0, "W_KILL": 250.0, "W_TAIL": 15.0,
        "W_CENTER": 5.0, "W_EDGE": 15.0, "W_CORNER": 40.0, "W_HAZARD": 1000.0,
        "W_GHOST": 50.0, "W_CONSTRICT": 15.0, "W_PIN": 200.0, "W_FOG_RISK": 35.0,
    },
    GamePhase.LATE_1V1: {
        "W_VORONOI": 15.0, "W_FOOD": 10.0, "W_KILL": 1500.0, "W_TAIL": 40.0,
        "W_CENTER": 0.0, "W_EDGE": 10.0, "W_CORNER": 25.0, "W_HAZARD": 1000.0,
        "W_GHOST": 70.0, "W_CONSTRICT": 55.0, "W_PIN": 1500.0, "W_FOG_RISK": 50.0,
    },
    GamePhase.LATE_FFA: {
        "W_VORONOI": 30.0, "W_FOOD": 20.0, "W_KILL": 400.0, "W_TAIL": 20.0,
        "W_CENTER": 0.0, "W_EDGE": 15.0, "W_CORNER": 40.0, "W_HAZARD": 1000.0,
        "W_GHOST": 50.0, "W_CONSTRICT": 0.0, "W_PIN": 0.0, "W_FOG_RISK": 30.0,
    },
}


def _load_weight_overrides() -> None:
    """If tune_weights.py has produced weights.json next to this file, merge
    it into PHASE_WEIGHTS at import time. Missing/malformed file -> no-op."""
    import json as _json
    import os as _os
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "weights.json")
    if not _os.path.isfile(path):
        return
    try:
        with open(path) as f:
            data = _json.load(f)
        for phase in GamePhase:
            if phase.value in data:
                PHASE_WEIGHTS[phase].update(data[phase.value])
        log.info(f"Loaded weight overrides from {path}")
    except Exception as e:
        log.warning(f"Failed to load weights.json: {e}")


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
        return int(data["game"]["ruleset"]["settings"]["viewRadius"])
    except (KeyError, TypeError, ValueError):
        return 5


def _get_hazard_dmg(data: dict) -> int:
    try:
        return int(data["game"]["ruleset"]["settings"]["hazardDamagePerTurn"])
    except (KeyError, TypeError, ValueError):
        return 0


def _is_in_view(px: int, py: int, hx: int, hy: int, radius: int) -> bool:
    return _manhattan(px, py, hx, hy) <= radius


def _new_mem_entry() -> dict:
    return {"food": set(), "enemy_info": {}, "food_meta": {}, "fallback_history": []}


# ===========================================================================
# Memory updates (required interfaces — signatures must not change)
# ===========================================================================
def _update_food_memory(game_id: str, data: dict) -> set[tuple[int, int]]:
    mem = _game_memory.setdefault(game_id, _new_mem_entry())
    prev_food: set = mem.setdefault("food", set())
    food_meta: dict = mem.setdefault("food_meta", {})

    you  = data["you"]
    head = you.get("head") or {}
    hx: int = head.get("x", 0)
    hy: int = head.get("y", 0)
    radius  = _get_view_radius(data)
    turn    = data.get("turn", 0)

    visible_food = {(f["x"], f["y"]) for f in data.get("board", {}).get("food", [])}

    new_memory: set = set()
    new_meta: dict = {}

    for pos in prev_food:
        px, py = pos
        if _is_in_view(px, py, hx, hy, radius):
            # In view: trust what we currently see, nothing more.
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
    mem = _game_memory.setdefault(game_id, _new_mem_entry())
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
                "length": s.get("length", 0),
            }

    # Purge dead / no-longer-present snakes.
    dead = set(enemy_info.keys()) - live_ids
    for sid in dead:
        del enemy_info[sid]

    return enemy_info


# ===========================================================================
# Engine Core
# ===========================================================================
class TacticalEngine:
    COMPUTE_BUDGET_S = COMPUTE_BUDGET_S
    DIRECTIONS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------
    @classmethod
    def get_best_move(cls, data: dict, merged_food: set, game_id: str, deadline: float) -> str:
        ctx = cls._build_context(data, merged_food, game_id, deadline)

        # [E] Opening book — turns 1-5, never fight, just rush food or center.
        if ctx.turn <= 5:
            opening = cls._get_opening_move(ctx)
            if opening:
                cls._record_history(game_id, ctx, "opening", len(cls.DIRECTIONS), opening, 0.0)
                return opening

        candidates = {
            d: (ctx.our_head[0] + dx, ctx.our_head[1] + dy)
            for d, (dx, dy) in cls.DIRECTIONS.items()
        }

        # Tier 1 (preferred): fully safe — passes every Layer-1 rule,
        # including corridor trap + N-turn escape margin.
        strict_pool = [d for d, c in candidates.items() if cls._is_safe(c, ctx)]

        if strict_pool:
            pool, tier = strict_pool, "strict"
        else:
            # Tier 2 (graduated relaxation): drop the corridor/escape-margin
            # *preference* but keep every non-negotiable rule. This is the
            # fix for the fallback-storm bug — we still hand the scorer a
            # real, ranked set of options instead of guessing blind.
            relaxed_pool = [
                d for d, c in candidates.items()
                if not cls._is_certain_death(c, ctx)
                and not (c in ctx.hazard_set and ctx.our_health < HAZARD_SOFT_HEALTH)
            ]
            if relaxed_pool:
                pool, tier = relaxed_pool, "relaxed"
                log.warning(
                    f"Turn {ctx.turn}: no move clears the corridor/escape margin; "
                    f"relaxing to non-negotiable rules only. pool={relaxed_pool}"
                )
            else:
                # Tier 3: every direction fails even the non-negotiable
                # rules — a genuine, inescapable box. Rank by expected
                # survivability rather than avoidance alone (Bug 4 fix).
                move = cls._last_resort_move(candidates, ctx, game_id)
                cls._record_history(game_id, ctx, "last_resort", 0, move, 0.0)
                return move

        scored = sorted(
            ((cls._score_move(candidates[d], ctx), d) for d in pool),
            reverse=True,
        )
        best_score, best_move = scored[0]
        cls._record_history(game_id, ctx, tier, len(pool), best_move, best_score)

        log.info(
            f"Turn {ctx.turn} | tier={tier} pool={pool} -> {best_move} "
            f"(score={best_score:.1f}, phase={ctx.phase.value})"
        )
        return best_move

    @classmethod
    def _record_history(cls, game_id: str, ctx: "GameContext", tier: str,
                         pool_size: int, move: str, score: float) -> None:
        """[G] Fallback Forensics — keep a rolling window of the last 10
        decisions (pool size, chosen move, score, phase) so that if/when the
        engine ever does hit `_last_resort_move`, we have real context to
        debug from immediately, no log-spelunking required."""
        mem = _game_memory.setdefault(game_id, _new_mem_entry())
        history = mem.setdefault("fallback_history", [])
        history.append(
            f"T{ctx.turn}: tier={tier} safe={pool_size} chose={move} "
            f"score={score:.1f} phase={ctx.phase.value}"
        )
        if len(history) > 10:
            history.pop(0)

    # -----------------------------------------------------------------------
    # Context builder
    # -----------------------------------------------------------------------
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

        our_tail = our_body[-1] if (len(our_body) > 1 and our_body[-1]) else None

        occupied: set = set()
        for b in our_body[:-1]:
            if b:
                occupied.add(b)

        # Tail-vacating: the tail cell is only solid if we just ate (server
        # signals this by duplicating the tail segment / health snaps to 100).
        if our_tail and our_health == 100:
            occupied.add(our_tail)

        enemy_data: list = []
        snakes = board.get("snakes", [])
        num_snakes = len(snakes)

        for s in snakes:
            if s["id"] == you["id"]:
                continue
            e_len = s.get("length", 1)
            e_body = [_pt(seg) for seg in s.get("body", [])]

            # Any visible body segment is a solid obstacle *regardless* of
            # whether this enemy's head specifically is currently visible.
            # (Confirmed bug: the previous version only added a snake's body
            # to `occupied` inside an `if e_head:` block. A long enemy's head
            # can drift outside our view radius while a trailing part of its
            # body is still plainly visible a step or two away — that body
            # segment was then silently dropped from `occupied`, and the
            # engine walked straight into a snake it could actually see.
            # Reproduced and confirmed against run_games.py --seed 101.)
            for b in e_body[:-1]:
                if b:
                    occupied.add(b)

            e_head = e_body[0] if e_body else None
            if e_head:
                e_health = s.get("health", 100)
                e_tail = e_body[-1] if (len(e_body) > 1 and e_body[-1]) else None
                if e_tail and e_health == 100:
                    occupied.add(e_tail)

                enemy_data.append({
                    "id": s["id"],
                    "head_pos": e_head,
                    "length": e_len,
                    "health": e_health,
                    "body": e_body,
                })

        hazard_set = {(f["x"], f["y"]) for f in board.get("hazards", [])}
        hazard_dmg = _get_hazard_dmg(data)

        # Board edges as solid occupied cells — lets every downstream check
        # (occupied lookup, flood-fill) treat OOB uniformly with body cells.
        for x in range(w):
            occupied.add((x, -1))
            occupied.add((x, h))
        for y in range(h):
            occupied.add((-1, y))
            occupied.add((w, y))

        view_radius = _get_view_radius(data)

        # Ghost zones — soft caution around a currently-unseen enemy's last
        # known head. Never a hard veto (spec: "consider it potentially
        # dangerous"), only a scoring penalty via W_GHOST.
        enemy_info = _game_memory.get(game_id, {}).get("enemy_info", {})
        ghost_zones: set = set()
        for einfo in enemy_info.values():
            if turn - einfo.get("last_seen_turn", 0) <= 0:
                continue  # currently visible, no ghost needed
            lh = einfo.get("last_head")
            if not lh:
                continue
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if abs(dx) + abs(dy) <= 2:
                        gx, gy = lh[0] + dx, lh[1] + dy
                        if 0 <= gx < w and 0 <= gy < h:
                            ghost_zones.add((gx, gy))

        unseen_cells: set = set()
        for x in range(w):
            for y in range(h):
                if not _is_in_view(x, y, our_head[0], our_head[1], view_radius):
                    unseen_cells.add((x, y))

        # [Phase Detector] 4 states.
        if turn < 20:
            phase = GamePhase.EARLY
        elif num_snakes == 2 and turn >= 20:
            phase = GamePhase.LATE_1V1
        elif num_snakes > 2 and turn > 60:
            phase = GamePhase.LATE_FFA
        else:
            phase = GamePhase.MID

        return GameContext(
            our_head=our_head, our_body=our_body, our_len=our_len,
            our_health=our_health, our_tail=our_tail,
            width=w, height=h, turn=turn, occupied=occupied,
            hazard_set=hazard_set, hazard_dmg=hazard_dmg,
            enemy_data=enemy_data,
            visible_food={(f["x"], f["y"]) for f in board.get("food", [])},
            merged_food=merged_food, phase=phase, weights=PHASE_WEIGHTS[phase],
            deadline=deadline, ghost_zones=ghost_zones, unseen_cells=unseen_cells,
        )

    # -----------------------------------------------------------------------
    # Layer 1: Survival Filter
    # -----------------------------------------------------------------------
    @classmethod
    def _is_certain_death(cls, cand: tuple[int, int], ctx: GameContext) -> bool:
        """Tier A — conditions with ~0% chance of survival. `occupied` already
        contains the OOB ring, so a single membership test covers both walls
        and body collisions (Bug fix: previously implicit, made explicit)."""
        if cand in ctx.occupied:
            return True

        # Lethal hazard: this step alone would take health to <= 0.
        if cand in ctx.hazard_set and (ctx.our_health - ctx.hazard_dmg) <= 0:
            return True

        # Head-to-head vs an enemy of equal or greater length: we lose or
        # both die. (Bug 1 fix: explicit parenthesisation — the original had
        # `nx != ep[0] or ny != ep[1]` unparenthesised next to an `and`,
        # which made the whole guard almost always True regardless of intent.)
        nx, ny = cand
        for e in ctx.enemy_data:
            eh = e["head_pos"]
            if eh and (_manhattan(nx, ny, eh[0], eh[1]) == 1) and (e["length"] >= ctx.our_len):
                return True

        return False

    @classmethod
    def _is_corridor_trap(cls, cand: tuple[int, int], ctx: GameContext) -> bool:
        """Tier B — is the immediate pocket already too small right now?"""
        required = ctx.our_len + CORRIDOR_MARGIN
        space = cls._flood_fill(cand, ctx.occupied, limit=required, deadline=ctx.deadline)
        return space < required

    @classmethod
    def _deep_escape_check(cls, cand: tuple[int, int], ctx: GameContext, depth: int = ESCAPE_DEPTH) -> bool:
        """[A] N-Turn Lookahead — simulate the board after 1..depth turns by
        relaxing tails (ours and every enemy's) toward their heads, and
        require the flood-fill space from `cand` to stay >= our_len +
        ESCAPE_MARGIN at every step. Prevents entering corridors that look
        wide now but pinch shut a few turns later."""
        for n in range(1, depth + 1):
            if time.monotonic() > ctx.deadline - 0.01:
                return False  # Bug 2 fix: assume danger on timeout, never "assume safe".

            sim_occupied: set = set()
            for x in range(ctx.width):
                sim_occupied.add((x, -1))
                sim_occupied.add((x, ctx.height))
            for y in range(ctx.height):
                sim_occupied.add((-1, y))
                sim_occupied.add((ctx.width, y))

            body_len = len(ctx.our_body)
            keep_len = max(1, body_len - n)
            for i in range(keep_len):
                if ctx.our_body[i]:
                    sim_occupied.add(ctx.our_body[i])

            for e in ctx.enemy_data:
                e_body = e["body"]
                e_blen = len(e_body)
                e_keep = max(1, e_blen - n)
                for i in range(e_keep):
                    if e_body[i]:
                        sim_occupied.add(e_body[i])

            required = ctx.our_len + ESCAPE_MARGIN
            space = cls._flood_fill(cand, sim_occupied, limit=required, deadline=ctx.deadline)
            if space < required:
                return False

        return True

    @classmethod
    def _is_safe(cls, cand: tuple[int, int], ctx: GameContext) -> bool:
        """Full Layer-1 check (strict tier): certain-death rules, the health
        gate on hazards, the corridor trap, and the N-turn escape check."""
        if cls._is_certain_death(cand, ctx):
            return False
        if cand in ctx.hazard_set and ctx.our_health < HAZARD_SOFT_HEALTH:
            return False
        if cls._is_corridor_trap(cand, ctx):
            return False
        if not cls._deep_escape_check(cand, ctx):
            return False
        return True

    @classmethod
    def _flood_fill(cls, start: tuple[int, int], occupied: set, limit: int, deadline: float) -> int:
        if start in occupied:
            return 0
        visited = {start}
        queue = deque([start])
        count = 0

        while queue:
            if count >= limit:
                break
            if time.monotonic() > deadline - 0.005:
                break  # timeout -> return partial (under-)count: conservative by construction.

            curr = queue.popleft()
            count += 1

            for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
                nxt = (curr[0] + dx, curr[1] + dy)
                if nxt not in occupied and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return count

    # -----------------------------------------------------------------------
    # Layer 2: Strategy Scorer
    # -----------------------------------------------------------------------
    @classmethod
    def _score_move(cls, cand: tuple[int, int], ctx: GameContext) -> float:
        w = ctx.weights

        # [F] Food Race — starvation override. Layer 1 has already filtered
        # out unsafe candidates, so this never trades survival for food; it
        # only decides *which* survivable move gets us to food fastest.
        if ctx.our_health < HUNGER_CRITICAL:
            food_dist = cls._bfs_dist(cand, ctx.merged_food, ctx.occupied, limit=50)
            if food_dist < 999:
                space = cls._flood_fill(cand, ctx.occupied, limit=ctx.our_len + ESCAPE_MARGIN, deadline=ctx.deadline)
                return -(food_dist * 100_000.0) + space

        score = 0.0

        # 1. Voronoi area — room to maneuver.
        v_area = cls._flood_fill(cand, ctx.occupied, limit=ctx.width * ctx.height, deadline=ctx.deadline)
        score += v_area * w["W_VORONOI"]

        # 2. Food proximity + [F] food race (non-critical health).
        food_dist = cls._bfs_dist(cand, ctx.merged_food, ctx.occupied, limit=50)
        if food_dist < 999:
            score -= food_dist * w["W_FOOD"]
            score += cls._food_race_score(cand, ctx)

        # 3. Kill opportunity — step adjacent to a strictly shorter enemy head.
        for e in ctx.enemy_data:
            eh = e["head_pos"]
            if eh and (e["length"] < ctx.our_len) and (_manhattan(cand[0], cand[1], eh[0], eh[1]) == 1):
                score += w["W_KILL"]

        # 4. [C] Coiling defense (tail-following).
        score += cls._coiling_score(cand, ctx)

        # 5. Positioning.
        cx, cy = ctx.width // 2, ctx.height // 2
        score -= _manhattan(cand[0], cand[1], cx, cy) * w["W_CENTER"]

        is_edge = (cand[0] == 0 or cand[0] == ctx.width - 1 or cand[1] == 0 or cand[1] == ctx.height - 1)
        is_corner = (cand[0] in (0, ctx.width - 1)) and (cand[1] in (0, ctx.height - 1))
        if is_edge:
            score -= w["W_EDGE"]
        if is_corner:
            score -= w["W_CORNER"]
        if cand in ctx.hazard_set:
            score -= w["W_HAZARD"]
        if cand in ctx.ghost_zones:
            score -= w["W_GHOST"]

        # 6. 1v1 tactics.
        if ctx.phase == GamePhase.LATE_1V1 and len(ctx.enemy_data) == 1:
            score += cls._constriction_score(cand, ctx)
            score += cls._pin_bonus(cand, ctx)
            score += cls._executioner_score(cand, ctx)

        # 7. [B] Fog-of-war conservative bias.
        score -= cls._fog_risk_score(cand, ctx)

        return score

    @classmethod
    def _bfs_dist(cls, start: tuple[int, int], targets: set, occupied: set, limit: int = 100) -> int:
        if not targets:
            return 999
        if start in targets:
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            curr, dist = queue.popleft()
            if dist > limit:
                break
            for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
                nxt = (curr[0] + dx, curr[1] + dy)
                if nxt in targets:
                    return dist + 1
                if nxt not in occupied and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
        return 999

    @classmethod
    def _food_race_score(cls, cand: tuple[int, int], ctx: GameContext) -> float:
        """[F] Food Race — reward being closer to food than every enemy;
        abandon a race we're already losing to an equal/longer snake."""
        if not ctx.merged_food or not ctx.enemy_data:
            return 0.0

        our_dist = cls._bfs_dist(cand, ctx.merged_food, ctx.occupied, limit=50)
        if our_dist >= 999:
            return 0.0

        target = min(ctx.merged_food, key=lambda f: _manhattan(cand[0], cand[1], f[0], f[1]))

        closest_enemy_dist = 999
        contender_len = 0
        for e in ctx.enemy_data:
            eh = e["head_pos"]
            if not eh:
                continue
            d = _manhattan(eh[0], eh[1], target[0], target[1])
            if d < closest_enemy_dist:
                closest_enemy_dist = d
                contender_len = e["length"]

        if our_dist < closest_enemy_dist:
            return ctx.weights["W_FOOD"]
        if (our_dist >= closest_enemy_dist) and (contender_len >= ctx.our_len):
            return -ctx.weights["W_FOOD"] * 10.0
        return 0.0

    @classmethod
    def _coiling_score(cls, cand: tuple[int, int], ctx: GameContext) -> float:
        """[C] Coiling Defense — when weaker or playing defensively, prefer
        moves that stay on the path back to our own tail."""
        if not ctx.our_tail:
            return 0.0

        longer_enemy_exists = any(e["length"] > ctx.our_len for e in ctx.enemy_data)
        defensive_posture = ctx.our_health > 40

        if not (longer_enemy_exists or defensive_posture):
            return 0.0
        if ctx.our_health < COIL_DISABLE_HEALTH:
            return 0.0
        if ctx.our_health < COIL_FOOD_HEALTH and any(
            _manhattan(cand[0], cand[1], f[0], f[1]) <= COIL_FOOD_DIST for f in ctx.visible_food
        ):
            return 0.0

        dist_to_tail = cls._bfs_dist(cand, {ctx.our_tail}, ctx.occupied, limit=30)
        if dist_to_tail < 999:
            return ctx.weights["W_TAIL"] * (30 - dist_to_tail)
        return 0.0

    @classmethod
    def _constriction_score(cls, cand: tuple[int, int], ctx: GameContext) -> float:
        """1v1 only — cheap Manhattan-distance proxy for "are we closing the
        gap on the enemy", used as a stand-in for a full Voronoi diff."""
        e = ctx.enemy_data[0]
        eh = e["head_pos"]
        if not eh:
            return 0.0
        dist_now = _manhattan(ctx.our_head[0], ctx.our_head[1], eh[0], eh[1])
        dist_next = _manhattan(cand[0], cand[1], eh[0], eh[1])
        return ctx.weights["W_CONSTRICT"] if dist_next < dist_now else 0.0

    @classmethod
    def _pin_bonus(cls, cand: tuple[int, int], ctx: GameContext) -> float:
        e = ctx.enemy_data[0]
        eh = e["head_pos"]
        if not eh or e["length"] >= ctx.our_len:
            return 0.0

        e_escapes = 0
        for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
            if (eh[0] + dx, eh[1] + dy) not in ctx.occupied:
                e_escapes += 1

        if (e_escapes == 1) and (_manhattan(cand[0], cand[1], eh[0], eh[1]) <= 2):
            return ctx.weights["W_PIN"]
        return 0.0

    @classmethod
    def _executioner_score(cls, cand: tuple[int, int], ctx: GameContext) -> float:
        """[D] 1v1 Executioner Mode — block a starving, shorter enemy's path
        to food."""
        e = ctx.enemy_data[0]
        if (e["length"] >= ctx.our_len) or (e["health"] >= 55):
            return 0.0
        eh = e["head_pos"]
        if not eh:
            return 0.0

        food_dist_for_enemy = cls._bfs_dist(eh, ctx.merged_food, ctx.occupied, limit=20)
        if food_dist_for_enemy < 999:
            our_dist_to_food = cls._bfs_dist(cand, ctx.merged_food, ctx.occupied, limit=20)
            blocking = (our_dist_to_food < food_dist_for_enemy) and (_manhattan(cand[0], cand[1], eh[0], eh[1]) == 1)
            if blocking:
                return ctx.weights["W_KILL"] * 2.0
        return 0.0

    @classmethod
    def _fog_risk_score(cls, cand: tuple[int, int], ctx: GameContext) -> float:
        """[B] Fog-of-war Conservative Bias — penalise (not veto) stepping
        into, or right next to, cells we currently can't see."""
        if cand in ctx.unseen_cells:
            return ctx.weights["W_FOG_RISK"]

        for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
            adj = (cand[0] + dx, cand[1] + dy)
            if (adj in ctx.unseen_cells) and (adj not in ctx.occupied):
                return ctx.weights["W_FOG_RISK"] * 0.5

        return 0.0

    # -----------------------------------------------------------------------
    # Opening & Fallback
    # -----------------------------------------------------------------------
    @classmethod
    def _get_opening_move(cls, ctx: GameContext) -> str | None:
        """[E] Turns 1-5: rush the closest safe visible food, else head to
        the center. Never fight — kill/pin/executioner scoring simply isn't
        consulted here."""
        best_dir = None
        best_dist = 999

        for d, (dx, dy) in cls.DIRECTIONS.items():
            cand = (ctx.our_head[0] + dx, ctx.our_head[1] + dy)
            if not cls._is_safe(cand, ctx):
                continue

            if ctx.visible_food:
                fd = cls._bfs_dist(cand, ctx.visible_food, ctx.occupied, limit=20)
                if fd < best_dist:
                    best_dist, best_dir = fd, d
            else:
                cx, cy = ctx.width // 2, ctx.height // 2
                cd = _manhattan(cand[0], cand[1], cx, cy)
                if cd < best_dist:
                    best_dist, best_dir = cd, d

        return best_dir

    @classmethod
    def _last_resort_move(cls, candidates: dict[str, tuple[int, int]], ctx: GameContext, game_id: str) -> str:
        """Bug 4 fix — invoked only when *every* direction fails even the
        non-negotiable Tier-A rules (a genuine, inescapable box). Rather than
        picking randomly or just avoiding `occupied`, rank all four raw
        directions by expected survivability: a contested head-to-head (the
        enemy might not actually move there) or a hazard tick we can still
        absorb both beat a guaranteed wall/body collision. Space is used as
        the final tie-break."""
        ranked = []
        for d, cand in candidates.items():
            nx, ny = cand
            risk = 0.0

            if cand in ctx.occupied:
                risk += 10_000.0  # unconditional death: wall or body

            if cand in ctx.hazard_set:
                resultant = ctx.our_health - ctx.hazard_dmg
                risk += 9_000.0 if resultant <= 0 else 200.0

            for e in ctx.enemy_data:
                eh = e["head_pos"]
                if eh and _manhattan(nx, ny, eh[0], eh[1]) == 1:
                    if e["length"] >= ctx.our_len:
                        risk += 500.0   # conditional death — enemy must also pick this cell
                    else:
                        risk -= 100.0   # a potential kill — mildly attractive, not risk

            space = cls._flood_fill(cand, ctx.occupied, limit=ctx.our_len + CORRIDOR_MARGIN, deadline=ctx.deadline)
            ranked.append((risk, -space, d))

        ranked.sort()
        best_move = ranked[0][2]

        history = _game_memory.setdefault(game_id, _new_mem_entry()).get("fallback_history", [])
        log.error(
            f"FALLBACK FORENSICS — Turn {ctx.turn}: every direction fails Tier-A safety. "
            f"Ranked(risk,-space,dir)={[(round(r, 0), s, dd) for r, s, dd in ranked]} "
            f"chosen={best_move}. Last 10 decisions: {history}"
        )
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
    _game_memory[game_id] = _new_mem_entry()
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


_load_weight_overrides()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
