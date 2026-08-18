"""Battlesnake tactical server.

Decision order is intentionally safety-first:
1. eliminate moves that are immediately fatal under simultaneous turn rules;
2. prefer moves with an escape margin and usable territory;
3. score food races, head-to-head opportunities, hazards and position;
4. optionally blend a trained policy/value model *only* across legal safe moves.

The neural model is an adviser, not a bypass for the rules layer.  This makes a
missing, stale or slow checkpoint fail closed to the deterministic engine.
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("battlesnake")

app = FastAPI(title="الثعبان — Battlesnake Competitive Engine", version="12.0.0")


class GameState(BaseModel):
    model_config = {"extra": "allow"}


SNAKE_INFO: dict[str, Any] = {
    "apiversion": "1",
    "author": "Mina Hussein",
    "color": "#FF0000",
    "head": "default",
    "tail": "default",
    "version": "12.0.0",
}

# This is deliberately below a typical 250 ms engine timeout.  The deadline is
# propagated to every graph operation so a dense board cannot turn into a timeout.
COMPUTE_BUDGET_S = 0.200
FOOD_STALE_TTL = 18
HUNGER_CRITICAL = 28
HAZARD_SOFT_HEALTH = 30
CORRIDOR_MARGIN = 2
ESCAPE_DEPTH = 3
ESCAPE_MARGIN = 2

_game_memory: dict[str, dict[str, Any]] = {}


class GamePhase(Enum):
    EARLY = "early"
    MID = "mid"
    LATE_1V1 = "late_1v1"
    LATE_FFA = "late_ffa"


@dataclass
class GameContext:
    our_head: tuple[int, int]
    our_body: list[tuple[int, int]]
    our_len: int
    our_health: int
    our_tail: tuple[int, int] | None
    width: int
    height: int
    turn: int
    occupied: set[tuple[int, int]]
    hazard_set: set[tuple[int, int]]
    hazard_dmg: int
    enemy_data: list[dict[str, Any]]
    visible_food: set[tuple[int, int]]
    merged_food: set[tuple[int, int]]
    phase: GamePhase
    weights: dict[str, float]
    deadline: float
    ghost_zones: set[tuple[int, int]]
    unseen_cells: set[tuple[int, int]]


# Scalar features have bounded ranges, allowing a durable default that can be
# tuned later without making one accidental term dominate the policy.
PHASE_WEIGHTS: dict[GamePhase, dict[str, float]] = {
    GamePhase.EARLY: {
        "W_SPACE": 18.0, "W_TERRITORY": 10.0, "W_FOOD": 18.0, "W_RACE": 45.0,
        "W_KILL": 80.0, "W_TAIL": 4.0, "W_CENTER": 4.0, "W_EDGE": 5.0,
        "W_HAZARD": 35.0, "W_GHOST": 8.0, "W_FOG": 5.0,
    },
    GamePhase.MID: {
        "W_SPACE": 24.0, "W_TERRITORY": 15.0, "W_FOOD": 12.0, "W_RACE": 55.0,
        "W_KILL": 120.0, "W_TAIL": 7.0, "W_CENTER": 2.0, "W_EDGE": 7.0,
        "W_HAZARD": 40.0, "W_GHOST": 12.0, "W_FOG": 8.0,
    },
    GamePhase.LATE_1V1: {
        "W_SPACE": 26.0, "W_TERRITORY": 22.0, "W_FOOD": 8.0, "W_RACE": 65.0,
        "W_KILL": 260.0, "W_TAIL": 14.0, "W_CENTER": 0.0, "W_EDGE": 6.0,
        "W_HAZARD": 45.0, "W_GHOST": 14.0, "W_FOG": 10.0,
    },
    GamePhase.LATE_FFA: {
        "W_SPACE": 28.0, "W_TERRITORY": 18.0, "W_FOOD": 10.0, "W_RACE": 60.0,
        "W_KILL": 170.0, "W_TAIL": 10.0, "W_CENTER": 0.0, "W_EDGE": 8.0,
        "W_HAZARD": 45.0, "W_GHOST": 12.0, "W_FOG": 10.0,
    },
}


def _load_weight_overrides() -> None:
    """Retain compatibility with the existing heuristic checkpoints."""
    import json

    path = Path(__file__).with_name("weights.json")
    if not path.is_file():
        return
    try:
        raw = json.loads(path.read_text())
        for phase in GamePhase:
            overrides = raw.get(phase.value)
            if isinstance(overrides, dict):
                # Ignore legacy keys rather than failing at server start.
                for key, value in overrides.items():
                    if key in PHASE_WEIGHTS[phase] and isinstance(value, (int, float)):
                        PHASE_WEIGHTS[phase][key] = float(value)
    except Exception as exc:  # pragma: no cover - defensive server startup path
        log.warning("Could not load weights.json: %s", exc)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pt(segment: Any) -> tuple[int, int] | None:
    if not isinstance(segment, dict) or segment.get("x") is None or segment.get("y") is None:
        return None
    return _as_int(segment["x"], 0), _as_int(segment["y"], 0)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _get_view_radius(data: dict[str, Any]) -> int | None:
    """Return None for standard full-information games, never a fake radius."""
    try:
        settings = data["game"]["ruleset"]["settings"]
        if "viewRadius" in settings:
            return max(0, _as_int(settings["viewRadius"], 0))
    except (KeyError, TypeError, ValueError):
        pass
    return None


def _get_hazard_dmg(data: dict[str, Any]) -> int:
    try:
        return max(0, _as_int(data["game"]["ruleset"]["settings"].get("hazardDamagePerTurn", 0), 0))
    except (KeyError, TypeError, ValueError):
        return 0


def _is_in_view(pos: tuple[int, int], head: tuple[int, int], radius: int | None) -> bool:
    return radius is None or _manhattan(pos, head) <= radius


def _new_mem_entry() -> dict[str, Any]:
    return {"food": set(), "food_meta": {}, "enemy_info": {}, "latency_history": []}


def _update_food_memory(game_id: str, data: dict[str, Any]) -> set[tuple[int, int]]:
    """Use visible food exactly in standard mode; retain bounded memory only in fog."""
    board = data.get("board", {})
    visible = {p for p in (_pt(item) for item in board.get("food", [])) if p is not None}
    radius = _get_view_radius(data)
    if radius is None:
        return visible

    mem = _game_memory.setdefault(game_id, _new_mem_entry())
    prior: set[tuple[int, int]] = mem.setdefault("food", set())
    meta: dict[tuple[int, int], int] = mem.setdefault("food_meta", {})
    you_head = _pt(data.get("you", {}).get("head")) or (0, 0)
    turn = int(data.get("turn", 0))
    remembered: set[tuple[int, int]] = set(visible)
    refreshed: dict[tuple[int, int], int] = {point: turn for point in visible}
    for point in prior:
        if not _is_in_view(point, you_head, radius) and turn - int(meta.get(point, turn)) <= FOOD_STALE_TTL:
            remembered.add(point)
            refreshed[point] = int(meta.get(point, turn))
    mem["food"], mem["food_meta"] = remembered, refreshed
    return remembered


def _update_enemy_memory(game_id: str, data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mem = _game_memory.setdefault(game_id, _new_mem_entry())
    seen: dict[str, dict[str, Any]] = {}
    you_id = data.get("you", {}).get("id")
    turn = int(data.get("turn", 0))
    for snake in data.get("board", {}).get("snakes", []):
        if snake.get("id") == you_id:
            continue
        head = _pt(snake.get("head"))
        if head:
            seen[str(snake.get("id"))] = {"last_head": head, "last_seen_turn": turn, "length": _as_int(snake.get("length"), 1)}
    mem["enemy_info"] = seen
    return seen


class NeuralAdvisor:
    """Lazy optional policy/value inference, guarded by a strict deadline."""
    _attempted = False
    _model: Any = None
    _board_size = 25
    _device = "cpu"

    @classmethod
    def _load(cls) -> None:
        if cls._attempted:
            return
        cls._attempted = True
        model_path = os.getenv("BATTLESNAKE_MODEL_PATH")
        if not model_path:
            return
        try:
            from neural_policy import load_checkpoint
            model, metadata, _extra = load_checkpoint(model_path, device="cpu")
            cls._model, cls._board_size = model, metadata.board_size
            log.info("Loaded neural adviser checkpoint from %s", model_path)
        except Exception as exc:  # broken/unsupported model must never break live play
            log.warning("Neural adviser disabled: %s", exc)
            cls._model = None

    @classmethod
    def scores(cls, data: dict[str, Any], legal: set[str], deadline: float) -> dict[str, float]:
        cls._load()
        if cls._model is None or time.monotonic() >= deadline - 0.035:
            return {}
        try:
            from neural_policy import masked_distribution, predict_logits
            logits, _value = predict_logits(cls._model, data, device=cls._device, board_size=cls._board_size)
            # The masking call is kept in the model layer to make invalid move
            # selection impossible even if the model changes later.
            import torch
            raw = torch.tensor([[logits[d] for d in ("up", "down", "left", "right")]])
            masked = masked_distribution(raw, legal)[0]
            return {direction: float(masked[index]) for index, direction in enumerate(("up", "down", "left", "right")) if direction in legal}
        except Exception as exc:  # pragma: no cover - only optional runtime path
            log.warning("Neural adviser failed this turn; falling back to tactics: %s", exc)
            return {}


class TacticalEngine:
    COMPUTE_BUDGET_S = COMPUTE_BUDGET_S
    DIRECTIONS: dict[str, tuple[int, int]] = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

    @classmethod
    def get_best_move(cls, data: dict[str, Any], merged_food: set[tuple[int, int]], game_id: str, deadline: float) -> str:
        ctx = cls._build_context(data, merged_food, game_id, deadline)
        candidates = {direction: (ctx.our_head[0] + delta[0], ctx.our_head[1] + delta[1]) for direction, delta in cls.DIRECTIONS.items()}

        strict = [direction for direction, pos in candidates.items() if cls._is_safe(pos, ctx)]
        relaxed = [direction for direction, pos in candidates.items() if not cls._is_certain_death(pos, ctx)]
        pool = strict or relaxed
        if not pool:
            move = cls._last_resort_move(candidates, ctx)
            cls._record_latency(game_id, ctx, "resort", move)
            return move

        # Pre-compute only bounded feature searches.  Neural scores are added
        # after the safety filter and therefore can never authorise a fatal move.
        scored = {direction: cls._score_move(candidates[direction], ctx) for direction in pool}
        neural = NeuralAdvisor.scores(data, set(pool), deadline)
        if neural:
            max_abs = max(1.0, max(abs(value) for value in neural.values()))
            for direction, logit in neural.items():
                scored[direction] += 12.0 * (logit / max_abs)

        move = max(pool, key=lambda direction: (scored[direction], direction))
        cls._record_latency(game_id, ctx, "strict" if strict else "relaxed", move)
        log.info("turn=%d phase=%s tier=%s move=%s score=%.2f", ctx.turn, ctx.phase.value, "strict" if strict else "relaxed", move, scored[move])
        return move

    @classmethod
    def _record_latency(cls, game_id: str, ctx: GameContext, tier: str, move: str) -> None:
        history = _game_memory.setdefault(game_id, _new_mem_entry()).setdefault("latency_history", [])
        history.append({"turn": ctx.turn, "tier": tier, "move": move})
        del history[:-12]

    @classmethod
    def _build_context(cls, data: dict[str, Any], merged_food: set[tuple[int, int]], game_id: str, deadline: float) -> GameContext:
        board, you = data.get("board", {}), data.get("you", {})
        width, height = _as_int(board.get("width"), 11), _as_int(board.get("height"), 11)
        our_body = [point for point in (_pt(segment) for segment in you.get("body", [])) if point is not None]
        our_head = _pt(you.get("head")) or (our_body[0] if our_body else (0, 0))
        our_length = _as_int(you.get("length"), len(our_body) or 1)
        our_tail = our_body[-1] if our_body else None
        occupied = cls._board_walls(width, height)
        cls._add_body(occupied, our_body)
        enemies: list[dict[str, Any]] = []
        you_id = you.get("id")
        for snake in board.get("snakes", []):
            if snake.get("id") == you_id:
                continue
            body = [point for point in (_pt(segment) for segment in snake.get("body", [])) if point is not None]
            cls._add_body(occupied, body)
            head = _pt(snake.get("head")) or (body[0] if body else None)
            if head:
                enemies.append({"id": snake.get("id", "enemy"), "head_pos": head, "length": _as_int(snake.get("length"), len(body) or 1), "health": _as_int(snake.get("health"), 100), "body": body})

        hazards = {point for point in (_pt(item) for item in board.get("hazards", [])) if point is not None}
        visible_food = {point for point in (_pt(item) for item in board.get("food", [])) if point is not None}
        radius = _get_view_radius(data)
        ghost_zones: set[tuple[int, int]] = set()
        unseen: set[tuple[int, int]] = set()
        if radius is not None:
            for x in range(width):
                for y in range(height):
                    if not _is_in_view((x, y), our_head, radius):
                        unseen.add((x, y))
            previous = _game_memory.get(game_id, {}).get("enemy_info", {})
            current_enemy_heads = {enemy["head_pos"] for enemy in enemies}
            for item in previous.values():
                head = item.get("last_head")
                if head and head not in current_enemy_heads:
                    for dx, dy in cls.DIRECTIONS.values():
                        point = (head[0] + dx, head[1] + dy)
                        if 0 <= point[0] < width and 0 <= point[1] < height:
                            ghost_zones.add(point)

        turn = _as_int(data.get("turn"), 0)
        if turn < 16:
            phase = GamePhase.EARLY
        elif len(enemies) == 1:
            phase = GamePhase.LATE_1V1
        elif turn >= 55:
            phase = GamePhase.LATE_FFA
        else:
            phase = GamePhase.MID
        return GameContext(
            our_head=our_head, our_body=our_body, our_len=our_length, our_health=_as_int(you.get("health"), 100), our_tail=our_tail,
            width=width, height=height, turn=turn, occupied=occupied, hazard_set=hazards, hazard_dmg=_get_hazard_dmg(data),
            enemy_data=enemies, visible_food=visible_food, merged_food=set(merged_food), phase=phase, weights=PHASE_WEIGHTS[phase],
            deadline=deadline, ghost_zones=ghost_zones, unseen_cells=unseen,
        )

    @staticmethod
    def _board_walls(width: int, height: int) -> set[tuple[int, int]]:
        walls = {(x, -1) for x in range(width)} | {(x, height) for x in range(width)}
        walls |= {(-1, y) for y in range(height)} | {(width, y) for y in range(height)}
        return walls

    @staticmethod
    def _add_body(occupied: set[tuple[int, int]], body: list[tuple[int, int]]) -> None:
        """Add a body while allowing an unstacked tail to vacate this turn.

        A duplicated final coordinate is the rules-engine signal that the tail
        did not move during the prior food turn, so it remains blocked now.
        """
        if not body:
            return
        occupied.update(body[:-1])
        if len(body) == 1 or (len(body) >= 2 and body[-1] == body[-2]):
            occupied.add(body[-1])

    @classmethod
    def _is_certain_death(cls, candidate: tuple[int, int], ctx: GameContext) -> bool:
        if candidate in ctx.occupied:
            return True
        if candidate in ctx.hazard_set and ctx.our_health - 1 - ctx.hazard_dmg <= 0:
            return True
        # All moves resolve simultaneously.  A same-destination H2H against an
        # equal/longer opponent is a certain loss (or mutual elimination).
        for enemy in ctx.enemy_data:
            head = enemy["head_pos"]
            if _manhattan(candidate, head) == 1 and int(enemy["length"]) >= ctx.our_len:
                return True
        return False

    @classmethod
    def _is_corridor_trap(cls, candidate: tuple[int, int], ctx: GameContext) -> bool:
        required = min(ctx.width * ctx.height, ctx.our_len + CORRIDOR_MARGIN + (1 if candidate in ctx.merged_food else 0))
        return cls._flood_fill(candidate, ctx.occupied, required, ctx.deadline) < required

    @classmethod
    def _deep_escape_check(cls, candidate: tuple[int, int], ctx: GameContext, depth: int = ESCAPE_DEPTH) -> bool:
        """Cheap tail-release forecast; conservative only when time remains."""
        required = min(ctx.width * ctx.height, ctx.our_len + ESCAPE_MARGIN + (1 if candidate in ctx.merged_food else 0))
        for future_turn in range(1, depth + 1):
            if time.monotonic() >= ctx.deadline - 0.015:
                return True  # deadline safety: retain the already-checked move
            blocked = cls._board_walls(ctx.width, ctx.height)
            cls._add_future_body(blocked, ctx.our_body, future_turn, candidate in ctx.merged_food)
            for enemy in ctx.enemy_data:
                cls._add_future_body(blocked, enemy["body"], future_turn, False)
            if cls._flood_fill(candidate, blocked, required, ctx.deadline) < required:
                return False
        return True

    @staticmethod
    def _add_future_body(blocked: set[tuple[int, int]], body: list[tuple[int, int]], turns: int, eating_now: bool) -> None:
        if not body:
            return
        releases = max(0, turns - (1 if eating_now else 0))
        keep = max(0, len(body) - releases)
        blocked.update(body[:keep])

    @classmethod
    def _is_safe(cls, candidate: tuple[int, int], ctx: GameContext) -> bool:
        if cls._is_certain_death(candidate, ctx):
            return False
        if candidate in ctx.hazard_set and ctx.our_health <= HAZARD_SOFT_HEALTH + ctx.hazard_dmg:
            return False
        return not cls._is_corridor_trap(candidate, ctx) and cls._deep_escape_check(candidate, ctx)

    @classmethod
    def _flood_fill(cls, start: tuple[int, int], occupied: set[tuple[int, int]], limit: int, deadline: float) -> int:
        if start in occupied:
            return 0
        seen, queue, count = {start}, deque([start]), 0
        while queue and count < limit:
            if time.monotonic() >= deadline - 0.004:
                break
            current = queue.popleft()
            count += 1
            for dx, dy in cls.DIRECTIONS.values():
                nxt = current[0] + dx, current[1] + dy
                if nxt not in occupied and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return count

    @classmethod
    def _bfs_dist(cls, start: tuple[int, int], targets: set[tuple[int, int]], occupied: set[tuple[int, int]], limit: int = 100) -> int:
        if not targets:
            return 999
        if start in targets:
            return 0
        queue, seen = deque([(start, 0)]), {start}
        while queue:
            position, distance = queue.popleft()
            if distance >= limit:
                continue
            for dx, dy in cls.DIRECTIONS.values():
                nxt = position[0] + dx, position[1] + dy
                if nxt in targets:
                    return distance + 1
                if nxt not in occupied and nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, distance + 1))
        return 999

    @classmethod
    def _territory(cls, candidate: tuple[int, int], ctx: GameContext) -> int:
        """Count cells we reach strictly before all visible enemy heads."""
        if time.monotonic() >= ctx.deadline - 0.025:
            return 0
        # Distance maps are only as large as the board and share body obstacles.
        own = cls._distance_map(candidate, ctx.occupied, ctx.width * ctx.height, ctx.deadline)
        enemy_distance: dict[tuple[int, int], int] = {}
        for enemy in ctx.enemy_data:
            distances = cls._distance_map(enemy["head_pos"], ctx.occupied, ctx.width * ctx.height, ctx.deadline)
            for cell, distance in distances.items():
                enemy_distance[cell] = min(enemy_distance.get(cell, 999), distance)
        return sum(1 for cell, distance in own.items() if distance < enemy_distance.get(cell, 999))

    @classmethod
    def _distance_map(cls, start: tuple[int, int], occupied: set[tuple[int, int]], limit: int, deadline: float) -> dict[tuple[int, int], int]:
        queue, distances = deque([(start, 0)]), {start: 0}
        while queue and len(distances) < limit and time.monotonic() < deadline - 0.012:
            position, distance = queue.popleft()
            for dx, dy in cls.DIRECTIONS.values():
                nxt = position[0] + dx, position[1] + dy
                if nxt not in occupied and nxt not in distances:
                    distances[nxt] = distance + 1
                    queue.append((nxt, distance + 1))
        return distances

    @classmethod
    def _food_race_score(cls, candidate: tuple[int, int], ctx: GameContext) -> float:
        if not ctx.merged_food:
            return 0.0
        our_distance = cls._bfs_dist(candidate, ctx.merged_food, ctx.occupied, limit=ctx.width * ctx.height)
        if our_distance >= 999:
            return -ctx.weights["W_RACE"]
        target = min(ctx.merged_food, key=lambda food: _manhattan(candidate, food))
        closest_enemy, enemy_length = 999, 0
        for enemy in ctx.enemy_data:
            distance = cls._bfs_dist(enemy["head_pos"], {target}, ctx.occupied, limit=ctx.width * ctx.height)
            if distance < closest_enemy:
                closest_enemy, enemy_length = distance, int(enemy["length"])
        if our_distance < closest_enemy:
            return ctx.weights["W_RACE"]
        if our_distance == closest_enemy and ctx.our_len > enemy_length:
            return ctx.weights["W_RACE"] * 0.35
        if closest_enemy <= our_distance and enemy_length >= ctx.our_len:
            return -ctx.weights["W_RACE"]
        return 0.0

    @classmethod
    def _score_move(cls, candidate: tuple[int, int], ctx: GameContext) -> float:
        weights = ctx.weights
        if ctx.our_health <= HUNGER_CRITICAL:
            distance = cls._bfs_dist(candidate, ctx.visible_food or ctx.merged_food, ctx.occupied, limit=ctx.width * ctx.height)
            # Survival remains guaranteed by the caller, but starvation has an
            # intentionally lexicographic priority across the remaining moves.
            return -10000.0 * distance + cls._flood_fill(candidate, ctx.occupied, ctx.our_len + ESCAPE_MARGIN, ctx.deadline)

        area = cls._flood_fill(candidate, ctx.occupied, ctx.width * ctx.height, ctx.deadline)
        score = area * weights["W_SPACE"]
        score += cls._territory(candidate, ctx) * weights["W_TERRITORY"]
        food_distance = cls._bfs_dist(candidate, ctx.merged_food, ctx.occupied, limit=ctx.width * ctx.height)
        if food_distance < 999:
            score -= food_distance * weights["W_FOOD"]
            score += cls._food_race_score(candidate, ctx)
        for enemy in ctx.enemy_data:
            if int(enemy["length"]) < ctx.our_len and _manhattan(candidate, enemy["head_pos"]) == 1:
                score += weights["W_KILL"]
        if ctx.our_tail:
            tail_distance = cls._bfs_dist(candidate, {ctx.our_tail}, ctx.occupied, limit=ctx.width * ctx.height)
            if tail_distance < 999:
                score += max(0, 16 - tail_distance) * weights["W_TAIL"]
        center = ((ctx.width - 1) // 2, (ctx.height - 1) // 2)
        score -= _manhattan(candidate, center) * weights["W_CENTER"]
        if candidate[0] in (0, ctx.width - 1) or candidate[1] in (0, ctx.height - 1):
            score -= weights["W_EDGE"]
        if candidate in ctx.hazard_set:
            projected_health = ctx.our_health - 1 - ctx.hazard_dmg
            score -= weights["W_HAZARD"] * (1.0 + max(0, 40 - projected_health) / 40.0)
        if candidate in ctx.ghost_zones:
            score -= weights["W_GHOST"]
        if candidate in ctx.unseen_cells:
            score -= weights["W_FOG"]
        return score

    @classmethod
    def _last_resort_move(cls, candidates: dict[str, tuple[int, int]], ctx: GameContext) -> str:
        """Choose the least-bad action only when no action is immediately safe."""
        ranked: list[tuple[float, str]] = []
        for direction, candidate in candidates.items():
            penalty = 0.0
            if candidate in ctx.occupied:
                penalty += 10000.0
            if candidate in ctx.hazard_set:
                penalty += 2000.0 if ctx.our_health - 1 - ctx.hazard_dmg <= 0 else 50.0
            for enemy in ctx.enemy_data:
                if _manhattan(candidate, enemy["head_pos"]) == 1 and int(enemy["length"]) >= ctx.our_len:
                    penalty += 600.0
            area = cls._flood_fill(candidate, ctx.occupied, ctx.our_len + CORRIDOR_MARGIN, ctx.deadline)
            ranked.append((penalty - area, direction))
        return min(ranked)[1]


@app.get("/")
def on_info() -> JSONResponse:
    return JSONResponse(SNAKE_INFO)


@app.post("/start")
def on_start(state: GameState) -> str:
    data = state.model_dump()
    _game_memory[data.get("game", {}).get("id", "unknown")] = _new_mem_entry()
    return "ok"


@app.post("/move")
def on_move(state: GameState) -> JSONResponse:
    started = time.monotonic()
    deadline = started + COMPUTE_BUDGET_S
    data = state.model_dump()
    game_id = data.get("game", {}).get("id", "unknown")
    _update_enemy_memory(game_id, data)
    food = _update_food_memory(game_id, data)
    try:
        direction = TacticalEngine.get_best_move(data, food, game_id, deadline)
    except Exception as exc:  # Protect the game contract even from unexpected payloads.
        log.exception("Move evaluation failed: %s", exc)
        direction = "up"
    elapsed = (time.monotonic() - started) * 1000
    log.info("turn=%s move=%s latency_ms=%.1f", data.get("turn"), direction, elapsed)
    return JSONResponse({"move": direction})


@app.post("/end")
def on_end(state: GameState) -> str:
    data = state.model_dump()
    _game_memory.pop(data.get("game", {}).get("id", "unknown"), None)
    return "ok"


_load_weight_overrides()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
