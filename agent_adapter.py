"""
agent_adapter.py — wraps our TacticalEngine as a bs-blackout-starter BaseAgent.

Exports:
  ThuebanAgent  — our main engine
  move_latency_ms — module-level list; each move() call appends its elapsed_ms
                    so run_games.py can collect per-move latency stats.
"""

from __future__ import annotations

import sys
import os
import time
from typing import List

# Project root importable
sys.path.insert(0, os.path.dirname(__file__))

from main import (
    TacticalEngine,
    _game_memory,
    _update_food_memory,
    _update_enemy_memory,
)

# ---------------------------------------------------------------------------
# Shared latency collector — filled by ThuebanAgent.move() each call.
# run_games.py reads this after each batch.
# ---------------------------------------------------------------------------
move_latency_ms: List[float] = []


class ThuebanAgent:
    """
    Adapter that wraps TacticalEngine into the bs-blackout-starter BaseAgent
    interface so it can participate in local hisss-driven games.
    """

    def get_name(self) -> str:
        return "الثعبان"

    def get_color(self) -> str:
        return "#1a1a2e"

    def get_author(self) -> str:
        return "Blackout2026"

    @staticmethod
    def _to_raw(game_state) -> dict:
        return game_state.model_dump() if hasattr(game_state, "model_dump") else game_state

    def start(self, game_state) -> None:
        data    = self._to_raw(game_state)
        game_id = data.get("game", {}).get("id", "unknown")
        _game_memory[game_id] = {"food": set(), "enemy_info": {}}

    def move(self, game_state):
        data    = self._to_raw(game_state)
        game_id = data.get("game", {}).get("id", "unknown")

        t0       = time.monotonic()
        deadline = t0 + TacticalEngine.COMPUTE_BUDGET_S

        _update_enemy_memory(game_id, data)
        merged_food = _update_food_memory(game_id, data)

        direction = TacticalEngine.get_best_move(
            data, merged_food, game_id=game_id, deadline=deadline
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        move_latency_ms.append(elapsed_ms)   # ← latency collector

        print(
            f"  [الثعبان] Turn {data.get('turn', 0):3d}  "
            f"{direction:5s}  {elapsed_ms:.1f} ms"
        )

        try:
            from battlesnake_types import MoveAction, Direction
            dir_map = {
                "up":    Direction.UP,
                "down":  Direction.DOWN,
                "left":  Direction.LEFT,
                "right": Direction.RIGHT,
            }
            return MoveAction(move=dir_map[direction])
        except ImportError:
            return direction

    def end(self, game_state) -> None:
        data    = self._to_raw(game_state)
        game_id = data.get("game", {}).get("id", "unknown")
        _game_memory.pop(game_id, None)
