"""Bounded no-tail-release path search for Battlesnake Constrictor.

It estimates future *self* mobility, not guaranteed survival against opponents.
Opponent head reachability is a pressure feature, not a predicted move.
Every path extension permanently occupies a square, matching Constrictor.
"""
from __future__ import annotations

import time
from collections import deque

_STEPS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def route_quality(
    start: tuple[int, int], *, width: int, height: int,
    occupied: set[tuple[int, int]],
    opponent_heads: tuple[tuple[int, int], ...] = (),
    max_depth: int = 18, beam_width: int = 72,
    deadline: float = float("inf"),
) -> tuple[int, int, int]:
    """Return (longest plausible route, uncontested path cells, exits).

    Depth counts the proposed move as step one. This is an intentionally
    bounded beam approximation of a longest self-avoiding route (NP-hard),
    not proof of a winning strategy or an opponent's future actions.
    """
    if width < 1 or height < 1 or width * height > 625:
        return (0, 0, 0)
    sx, sy = start
    if not (0 <= sx < width and 0 <= sy < height) or start in occupied:
        return (0, 0, 0)

    size = width * height
    adjacent: list[tuple[int, ...]] = []
    for idx in range(size):
        x, y = idx % width, idx // width
        adjacent.append(tuple((x + dx) + width * (y + dy)
                              for dx, dy in _STEPS
                              if 0 <= x + dx < width and 0 <= y + dy < height))
    static = 0
    for x, y in occupied:
        if 0 <= x < width and 0 <= y < height:
            static |= 1 << (x + width * y)

    # A static-body estimate of when an enemy could first reach a free cell.
    # Opponents still act simultaneously; this map is not a guarantee.
    arrival = [size + 1] * size
    queue = deque()
    for x, y in opponent_heads:
        if 0 <= x < width and 0 <= y < height:
            idx = x + width * y
            arrival[idx] = 0
            queue.append(idx)
    while queue:
        pos = queue.popleft()
        for nxt in adjacent[pos]:
            if (static & (1 << nxt)) or arrival[nxt] <= arrival[pos] + 1:
                continue
            arrival[nxt] = arrival[pos] + 1
            queue.append(nxt)

    first = sx + width * sy
    safe = int(arrival[first] > 1)
    # (head index, permanent occupied mask, uncontested path count)
    beam = [(first, static | (1 << first), safe)]
    best = (1, safe, sum(not (beam[0][1] & (1 << p)) for p in adjacent[first]))
    for depth in range(2, min(max_depth, size) + 1):
        if time.monotonic() >= deadline - 0.018:
            break
        future: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int]] = set()
        for head, mask, safe_count in beam:
            for nxt in adjacent[head]:
                bit = 1 << nxt
                if mask & bit:
                    continue
                new_mask = mask | bit
                key = (nxt, new_mask)
                if key in seen:
                    continue
                seen.add(key)
                new_safe = safe_count + int(arrival[nxt] > depth)
                exits = sum(not (new_mask & (1 << neighbor)) for neighbor in adjacent[nxt])
                future.append((nxt, new_mask, new_safe, exits))
        if not future:
            break
        # Depth dominates the final score; use pressure/exits to prune.
        future.sort(key=lambda v: (v[2], v[3], -v[0]), reverse=True)
        head, mask, safe_count, exits = future[0]
        best = (depth, safe_count, exits)
        beam = [(p, bits, safety) for p, bits, safety, _ in future[:beam_width]]
    return best
