"""Small, deterministic survival search for Battlesnake's simultaneous-turn rules.

This module has no web-server or neural-network dependencies. It deliberately
models our own body exactly, while treating other bodies as stationary for the
short horizon; it does not pretend to predict an opponent's policy.
"""
from __future__ import annotations

import heapq
import time
from typing import Any

DIRECTIONS = ((0, 1), (0, -1), (-1, 0), (1, 0))


def game_mode(data: dict[str, Any]) -> str:
    ruleset = data.get("game", {}).get("ruleset", {})
    return str(ruleset.get("name", "standard")).lower()


def is_constrictor(mode: str) -> bool:
    return "constrictor" in mode


def is_wrapped(mode: str) -> bool:
    return "wrapped" in mode


def neighbor(point: tuple[int, int], step: tuple[int, int], width: int, height: int, mode: str) -> tuple[int, int]:
    x, y = point[0] + step[0], point[1] + step[1]
    if is_wrapped(mode):
        return x % width, y % height
    return x, y


def in_bounds(point: tuple[int, int], width: int, height: int) -> bool:
    return 0 <= point[0] < width and 0 <= point[1] < height


def opponent_destinations(enemy: dict[str, Any], occupied: set[tuple[int, int]], width: int, height: int, mode: str) -> set[tuple[int, int]]:
    """Destinations an opponent could legally enter, ignoring unobserved choices."""
    head = enemy["head_pos"]
    return {
        point for step in DIRECTIONS
        if in_bounds(point := neighbor(head, step, width, height, mode), width, height)
        and point not in occupied
    }


def head_to_head_risk(candidate: tuple[int, int], enemies: list[dict[str, Any]], occupied: set[tuple[int, int]], our_length: int, width: int, height: int, mode: str) -> int:
    """Return 2 for potential lethal tie/loss, 1 for potentially winning contest, else 0.

    Risk is *possible*, not certain: all snakes choose their moves simultaneously.
    """
    risk = 0
    for enemy in enemies:
        if candidate in opponent_destinations(enemy, occupied, width, height, mode):
            risk = max(risk, 2 if int(enemy["length"]) >= our_length else 1)
    return risk


def move_body(body: tuple[tuple[int, int], ...], position: tuple[int, int], food: frozenset[tuple[int, int]], health: int, hazards: set[tuple[int, int]], hazard_damage: int, static_occupied: set[tuple[int, int]], width: int, height: int, mode: str) -> tuple[tuple[tuple[int, int], ...], frozenset[tuple[int, int]], int] | None:
    """Apply one single-snake turn with exact tail retention and food/health order."""
    if not in_bounds(position, width, height) or position in static_occupied:
        return None
    eats = position in food and not is_constrictor(mode)
    grows = eats or is_constrictor(mode)
    # A tail is traversable only if it moves THIS turn. A stacked tail is still
    # caught by body[:-1] when moving onto its old location.
    if position in (body if grows else body[:-1]):
        return None
    new_health = 100 if eats or is_constrictor(mode) else health - 1 - (hazard_damage if position in hazards else 0)
    if new_health <= 0:
        return None
    next_body = (position, *body) if grows else (position, *body[:-1])
    next_food = food - {position} if eats else food
    return next_body, next_food, new_health


def survive_horizon(candidate: tuple[int, int], ctx: Any, depth: int = 5) -> bool:
    """Existential bounded self-play: reject proven traps, not uncertain states.

    Opponent bodies are treated as blockers; opponent futures are deliberately
    not simulated. On deadline exhaustion, give the candidate the benefit of doubt.
    """
    mode = ctx.game_mode
    body = tuple(ctx.our_body)
    if not body:
        return True
    static = ctx.occupied.difference(body)
    foods = frozenset(ctx.visible_food)
    initial = move_body(body, candidate, foods, ctx.our_health, ctx.hazard_set,
                        ctx.hazard_dmg, static, ctx.width, ctx.height, mode)
    if initial is None:
        return False
    if depth <= 1:
        return True
    seen: set[tuple[Any, ...]] = set()

    def possible(state: tuple[Any, ...], remaining: int) -> bool:
        if remaining == 0 or time.monotonic() >= ctx.deadline - 0.012:
            return True
        snake, food, health = state
        key = (snake, food, health, remaining)
        if key in seen:
            return False
        seen.add(key)
        for step in DIRECTIONS:
            target = neighbor(snake[0], step, ctx.width, ctx.height, mode)
            next_state = move_body(snake, target, food, health, ctx.hazard_set,
                                   ctx.hazard_dmg, static, ctx.width, ctx.height, mode)
            if next_state is not None and possible(next_state, remaining - 1):
                return True
        return False

    return possible(initial, depth - 1)


def safe_food_distance(start: tuple[int, int], food: set[tuple[int, int]], blocked: set[tuple[int, int]], hazards: set[tuple[int, int]], health: int, hazard_damage: int, width: int, height: int, mode: str) -> int | None:
    """Shortest survivable route to food, accounting for accumulated hazard cost.

    Returns None if no reachable food exists before starvation. This is a
    one-snake path heuristic, NOT proof a moving opponent cannot intercept.
    """
    if not food:
        return None
    if start in food:
        return 0
    if health <= 0:
        return None
    frontier = [(0, -health, start)]
    best: dict[tuple[tuple[int, int], int], int] = {(start, health): 0}
    while frontier:
        steps, neg_health, point = heapq.heappop(frontier)
        if steps != best.get((point, -neg_health)):
            continue
        for step in DIRECTIONS:
            dest = neighbor(point, step, width, height, mode)
            if not in_bounds(dest, width, height) or (dest in blocked and dest not in food):
                continue
            if dest in food:
                return steps + 1
            left = -neg_health - 1 - (hazard_damage if dest in hazards else 0)
            if left <= 0:
                continue
            next_cost = (steps + 1, -left)
            key = (dest, left)
            if steps + 1 < best.get(key, 10**9):
                best[key] = steps + 1
                heapq.heappush(frontier, (*next_cost, dest))
    return None
