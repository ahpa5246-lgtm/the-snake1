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
        # Simultaneous head-to-head is a possible loss, not an inevitable one.
        self.assertFalse(main.TacticalEngine._is_certain_death((6, 5), context))
        self.assertEqual(main.TacticalEngine._head_risk((6, 5), context), 2)
        self.assertFalse(main.TacticalEngine._is_safe((6, 5), context))

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


class DeploymentParityTests(unittest.TestCase):
    def setUp(self):
        main._game_memory.clear()
        self.raw = materialize(state(
            body=[{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}],
            food=[{"x": 6, "y": 5}], turn=20,
        ))

    def test_tactical_policy_accepts_an_isolated_frozen_adviser(self):
        class Advisor:
            @staticmethod
            def scores(raw, legal, deadline):
                return {direction: (20.0 if direction == "left" else 0.0) for direction in legal}

        try:
            direction = main.TacticalEngine.get_best_move(
                self.raw, {(6, 5)}, "isolated", time.monotonic() + main.COMPUTE_BUDGET_S,
                advisor=Advisor,
            )
        except TypeError:
            self.fail("The deployed engine cannot evaluate isolated candidate/baseline advisers")
        self.assertIn(direction, main.TacticalEngine.DIRECTIONS)

    def test_frozen_adviser_still_cannot_bypass_immediate_death_filter(self):
        raw = materialize(state(
            body=[{"x": 0, "y": 5}, {"x": 0, "y": 4}, {"x": 0, "y": 3}],
        ))

        class Advisor:
            @staticmethod
            def scores(raw, legal, deadline):
                return {direction: 1e6 for direction in legal}

        try:
            direction = main.TacticalEngine.get_best_move(
                raw, set(), "safety", time.monotonic() + main.COMPUTE_BUDGET_S, advisor=Advisor,
            )
        except TypeError:
            self.fail("Frozen adviser evaluation is unavailable")
        self.assertNotEqual(direction, "left")
        self.assertNotEqual(direction, "down")

    def test_evaluator_uses_the_same_tactical_neural_policy_as_move_endpoint(self):
        from unittest.mock import patch
        from training import evaluate_candidate as evaluation
        agent_type = getattr(evaluation, "_DeployedPolicyAgent", None)
        self.assertIsNotNone(agent_type, "Evaluation currently measures a different policy from production")
        agent = agent_type(None, 25, "evaluation")

        class Advisor:
            @staticmethod
            def scores(raw, legal, deadline):
                return {direction: (30.0 if direction == "left" else -1.0) for direction in legal}

        agent.advisor = Advisor
        with patch.object(main.NeuralAdvisor, "scores", side_effect=Advisor.scores):
            production = json.loads(main.on_move(main.GameState.model_validate(self.raw)).body)["move"]
        self.assertEqual(agent.choose(self.raw), production)

    def test_candidate_and_baseline_advisers_do_not_share_model_state(self):
        from training import evaluate_candidate as evaluation
        agent_type = getattr(evaluation, "_DeployedPolicyAgent", None)
        self.assertIsNotNone(agent_type, "No production-policy evaluation adapter exists")
        original = main.NeuralAdvisor._model
        candidate_model, baseline_model = object(), object()
        candidate = agent_type(candidate_model, 25, "candidate")
        baseline = agent_type(baseline_model, 25, "baseline")
        self.assertIs(candidate.advisor._model, candidate_model)
        self.assertIs(baseline.advisor._model, baseline_model)
        self.assertIs(main.NeuralAdvisor._model, original)

    def test_frozen_policy_records_decisions_that_exceed_the_live_budget(self):
        from unittest.mock import patch
        from training.evaluate_candidate import _DeployedPolicyAgent
        agent = _DeployedPolicyAgent(None, 25, "slow-evaluation")
        clock = [0.0]

        def slow_decision(*args, **kwargs):
            clock[0] += main.COMPUTE_BUDGET_S + 0.01
            return "up"

        with patch("time.monotonic", side_effect=lambda: clock[0]):
            with patch.object(main.TacticalEngine, "get_best_move", side_effect=slow_decision):
                agent.choose(self.raw)
        self.assertEqual(getattr(agent, "deadline_overruns", None), 1)

    def test_late_decisions_cannot_pass_the_evaluation_gate(self):
        from training.evaluate_candidate import summarize_matches
        matches = []
        for candidate_slot, baseline_slot in ((0, 2), (2, 0)):
            matches.append({
                "seed": 7103, "outcome": "candidate_win",
                "candidate": {"slot": candidate_slot, "invalid_directions": 0,
                              "survival_turns": 100, "final_length": 8, "deadline_overruns": 1},
                "baseline": {"slot": baseline_slot, "invalid_directions": 0,
                             "survival_turns": 90, "final_length": 7, "deadline_overruns": 0},
            })
        result = summarize_matches(
            matches, baseline_model_sha256="a"*64, candidate_model_sha256="b"*64,
            baseline_checkpoint_sha256="c"*64, candidate_checkpoint_sha256="d"*64,
            seeds=[7103], minimum_games=2,
        )
        self.assertFalse(result["gate"]["passed"])
        self.assertIn("decision_budget_exceeded", result["gate"]["blockers"])

    def test_hidden_negative_coordinates_are_not_real_board_positions(self):
        self.assertIsNone(main._pt({"x": -1, "y": -1}))

    def test_hidden_enemy_heads_are_excluded_from_geometric_pathfinding(self):
        hidden = {"id": "hidden", "health": 100, "length": 3,
                  "head": {"x": -1, "y": -1},
                  "body": [{"x": -1, "y": -1}] * 3}
        raw = materialize(state(
            body=[{"x": 2, "y": 2}, {"x": 2, "y": 1}, {"x": 2, "y": 0}],
            enemies=[hidden], food=[{"x": 7, "y": 7}], width=15, height=15,
        ))
        context = main.TacticalEngine._build_context(raw, {(7, 7)}, "hidden", float("inf"))
        self.assertEqual(context.enemy_data, [])

    def test_food_race_on_partial_board_stays_inside_the_live_budget(self):
        hidden = [{"id": f"hidden-{i}", "health": 100, "length": 3,
                   "head": {"x": -1, "y": -1},
                   "body": [{"x": -1, "y": -1}] * 3} for i in range(3)]
        raw = materialize(state(
            body=[{"x": 2, "y": 2}, {"x": 2, "y": 1}, {"x": 2, "y": 0}],
            enemies=hidden, food=[{"x": 7, "y": 7}], width=15, height=15,
        ))
        started = time.monotonic()
        response = main.on_move(main.GameState.model_validate(raw))
        self.assertIn(json.loads(response.body)["move"], main.TacticalEngine.DIRECTIONS)
        self.assertLess(time.monotonic() - started, main.COMPUTE_BUDGET_S + 0.05)

    def test_bounded_bfs_rejects_an_outside_start(self):
        try:
            distance = main.TacticalEngine._bfs_dist(
                (-1, -1), {(7, 7)}, set(), width=15, height=15,
                deadline=time.monotonic() + 0.01,
            )
        except TypeError:
            self.fail("Distance search has no explicit board or deadline boundary")
        self.assertEqual(distance, 999)
