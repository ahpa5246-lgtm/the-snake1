from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.evaluate_candidate import (
    mirrored_slot_pairs,
    select_legal_action,
    sha256_model_state,
    summarize_matches,
    validate_held_out_seeds,
)


def match(
    seed: int,
    outcome: str,
    candidate_slot: int,
    baseline_slot: int,
    *,
    invalid: int = 0,
    failure: str | None = None,
) -> dict:
    return {
        "seed": seed,
        "outcome": outcome,
        "candidate": {
            "slot": candidate_slot,
            "invalid_directions": invalid,
            "survival_turns": 100,
            "final_length": 8,
            "failure_reason": failure,
        },
        "baseline": {
            "slot": baseline_slot,
            "invalid_directions": 0,
            "survival_turns": 90,
            "final_length": 7,
        },
    }


def paired_matches(seeds: list[int], outcomes: list[str] | None = None) -> list[dict]:
    expected = len(seeds) * 2
    results = outcomes or ["draw"] * expected
    if len(results) != expected:
        raise ValueError("one outcome is required for every mirrored game")
    matches = []
    outcome_index = 0
    for game_number, seed in enumerate(seeds):
        for candidate_slot, baseline_slot in mirrored_slot_pairs(game_number):
            matches.append(
                match(seed, results[outcome_index], candidate_slot, baseline_slot)
            )
            outcome_index += 1
    return matches


def summarize(matches: list[dict], seeds: list[int], *, same_model: bool = False) -> dict:
    return summarize_matches(
        matches,
        baseline_model_sha256="a" * 64,
        candidate_model_sha256=("a" if same_model else "b") * 64,
        baseline_checkpoint_sha256="c" * 64,
        candidate_checkpoint_sha256="d" * 64,
        seeds=seeds,
    )


class EvaluationGateTests(unittest.TestCase):
    def test_distinct_stronger_candidate_is_only_eligible_for_manual_review(self):
        seeds = list(range(7100, 7108))
        matches = paired_matches(
            seeds,
            ["candidate_win"] * 10 + ["draw"] * 4 + ["baseline_win"] * 2,
        )
        matches[-1]["candidate"]["failure_reason"] = "late_elimination"
        matches[-2]["candidate"]["failure_reason"] = "late_elimination"
        report = summarize(matches, seeds)
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(report["gate"]["recommendation"], "eligible_for_manual_review")
        self.assertEqual(report["matches_per_seed"], 2)
        self.assertFalse(report["promotion_performed"])

    def test_resaved_unchanged_model_keeps_frozen_baseline(self):
        seeds = list(range(7200, 7208))
        report = summarize(paired_matches(seeds), seeds, same_model=True)
        self.assertFalse(report["gate"]["passed"])
        self.assertTrue(report["checkpoints"]["distinct_files"])
        self.assertFalse(report["model_states"]["distinct"])
        self.assertIn("candidate_matches_baseline", report["gate"]["blockers"])

    def test_invalid_candidate_action_keeps_frozen_baseline(self):
        seeds = list(range(7250, 7258))
        matches = paired_matches(seeds, ["candidate_win"] * 10 + ["draw"] * 6)
        matches[0]["candidate"]["invalid_directions"] = 1
        report = summarize(matches, seeds)
        self.assertFalse(report["gate"]["passed"])
        self.assertEqual(report["gate"]["recommendation"], "retain_frozen_baseline")
        self.assertIn("invalid_directions_observed", report["gate"]["blockers"])

    def test_loss_profile_is_machine_readable(self):
        seeds = list(range(7300, 7308))
        matches = paired_matches(seeds)
        for index, failure in enumerate(
            ["early_elimination", "early_elimination", "starvation_or_hazard"]
        ):
            matches[index]["outcome"] = "baseline_win"
            matches[index]["candidate"]["failure_reason"] = failure
        report = summarize(matches, seeds)
        self.assertEqual(report["candidate"]["loss_profile"]["early_elimination"], 2)
        self.assertEqual(report["candidate"]["loss_profile"]["starvation_or_hazard"], 1)

    def test_every_seed_requires_swapped_candidate_and_baseline_slots(self):
        seeds = list(range(7400, 7408))
        matches = paired_matches(seeds)
        matches[1]["candidate"]["slot"] = matches[0]["candidate"]["slot"]
        matches[1]["baseline"]["slot"] = matches[0]["baseline"]["slot"]
        with self.assertRaisesRegex(ValueError, "missing a mirrored seat assignment"):
            summarize(matches, seeds)

    def test_slot_pair_helper_returns_exact_mirror(self):
        first, second = mirrored_slot_pairs(3)
        self.assertEqual(first, (3, 1))
        self.assertEqual(second, (1, 3))

    def test_engine_illegal_action_is_counted_and_replaced_deterministically(self):
        action, invalid = select_legal_action(
            "left",
            available_actions=["up", "right"],
            fallback_order=["up", "left", "right"],
        )
        self.assertEqual(action, "up")
        self.assertTrue(invalid)
        self.assertEqual(
            select_legal_action("right", ["up", "right"], ["up", "right"]),
            ("right", False),
        )

    def test_model_state_digest_is_content_based(self):
        self.assertEqual(
            sha256_model_state({"weight": b"same"}),
            sha256_model_state({"weight": b"same"}),
        )
        self.assertNotEqual(
            sha256_model_state({"weight": b"same"}),
            sha256_model_state({"weight": b"changed"}),
        )

    def test_training_seed_cannot_leak_into_held_out_set(self):
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            validate_held_out_seeds([2026, 7103], training_seed=2026)


if __name__ == "__main__":
    unittest.main()
