from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


def state(*, body, enemies=None, food=None, hazards=None, health=90, width=11, height=11, turn=10, settings=None):
    enemies = enemies or []
    return {
        "game": {"id": "test-game", "ruleset": {"settings": settings or {"hazardDamagePerTurn": 14}}},
        "turn": turn,
        "board": {"width": width, "height": height, "food": food or [], "hazards": hazards or [], "snakes": []},
        "you": {"id": "us", "health": health, "length": len(body), "head": body[0], "body": body},
    } | {"_enemies": enemies}


def materialize(raw):
    raw = dict(raw)
    enemies = raw.pop("_enemies", [])
    raw["board"] = dict(raw["board"])
    raw["board"]["snakes"] = [raw["you"], *enemies]
    return raw


class CompetitiveEngineTests(unittest.TestCase):
    def setUp(self):
        main._game_memory.clear()

    def test_full_information_keeps_only_visible_food(self):
        raw = materialize(state(body=[{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], food=[{"x": 6, "y": 5}]))
        remembered = main._update_food_memory("g", raw)
        self.assertEqual(remembered, {(6, 5)})
        self.assertIsNone(main._get_view_radius(raw))

    def test_unstacked_tail_is_passable_but_stacked_tail_is_not(self):
        raw = materialize(state(body=[{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 4, "y": 4}]))
        context = main.TacticalEngine._build_context(raw, set(), "g", float("inf"))
        self.assertFalse(main.TacticalEngine._is_certain_death((4, 4), context))
        raw["you"]["body"] = [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 4, "y": 4}, {"x": 4, "y": 4}]
        raw["you"]["length"] = 4
        raw["board"]["snakes"][0] = raw["you"]
        stacked = main.TacticalEngine._build_context(raw, set(), "g", float("inf"))
        self.assertTrue(main.TacticalEngine._is_certain_death((4, 4), stacked))

    def test_equal_length_head_to_head_is_filtered(self):
        enemy = {"id": "enemy", "health": 90, "length": 3, "head": {"x": 7, "y": 5}, "body": [{"x": 7, "y": 5}, {"x": 7, "y": 4}, {"x": 7, "y": 3}]}
        raw = materialize(state(body=[{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], enemies=[enemy]))
        context = main.TacticalEngine._build_context(raw, set(), "g", float("inf"))
        self.assertTrue(main.TacticalEngine._is_certain_death((6, 5), context))

    def test_terminal_death_reward_is_exact(self):
        before = materialize(state(body=[{"x": 5, "y": 5}], health=10))
        after = materialize(state(body=[{"x": 5, "y": 5}], health=0))
        after["you"]["body"] = []
        self.assertEqual(main_reward(before, after, alive=False, won=False), -1.0)

    def test_move_endpoint_returns_direction_within_budget(self):
        raw = materialize(state(body=[{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], food=[{"x": 6, "y": 5}], turn=20))
        started = time.monotonic()
        response = main.on_move(main.GameState.model_validate(raw))
        elapsed = time.monotonic() - started
        payload = json.loads(response.body)
        self.assertIn(payload["move"], main.TacticalEngine.DIRECTIONS)
        self.assertLess(elapsed, main.COMPUTE_BUDGET_S + 0.05)


def main_reward(before, after, *, alive, won):
    # Local import keeps production main.py independent from training extras.
    from training.neural_selfplay import transition_reward
    return transition_reward(before, after, alive_after=alive, won=won)


if __name__ == "__main__":
    unittest.main()
