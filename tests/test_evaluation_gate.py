from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.evaluate_candidate import summarize_matches, validate_held_out_seeds


def match(outcome: str, *, invalid: int = 0, failure: str | None = None) -> dict:
    return {
        "outcome": outcome,
        "candidate": {
            "invalid_directions": invalid,
            "survival_turns": 100,
            "final_length": 8,
            "failure_reason": failure,
        },
        "baseline": {
            "invalid_directions": 0,
            "survival_turns": 90,
            "final_length": 7,
        },
    }


class EvaluationGateTests(unittest.TestCase):
    def test_distinct_stronger_candidate_is_only_eligible_for_manual_review(self):
        seeds = list(range(7100, 7108))
        matches = [match("candidate_win") for _ in range(5)] + [
            match("draw"),
            match("draw"),
            match("baseline_win", failure="late_elimination"),
        ]
        report = summarize_matches(
            matches,
            baseline_sha256="a" * 64,
            candidate_sha256="b" * 64,
            seeds=seeds,
        )
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(report["gate"]["recommendation"], "eligible_for_manual_review")
        self.assertFalse(report["promotion_performed"])

    def test_equal_or_invalid_candidate_keeps_frozen_baseline(self):
        seeds = list(range(7200, 7208))
        matches = [match("candidate_win", invalid=1)] + [match("draw") for _ in range(7)]
        report = summarize_matches(
            matches,
            baseline_sha256="a" * 64,
            candidate_sha256="a" * 64,
            seeds=seeds,
        )
        self.assertFalse(report["gate"]["passed"])
        self.assertEqual(report["gate"]["recommendation"], "retain_frozen_baseline")
        self.assertIn("candidate_matches_baseline", report["gate"]["blockers"])
        self.assertIn("invalid_directions_observed", report["gate"]["blockers"])

    def test_loss_profile_is_machine_readable(self):
        seeds = list(range(7300, 7308))
        matches = [
            match("baseline_win", failure="early_elimination"),
            match("baseline_win", failure="early_elimination"),
            match("baseline_win", failure="starvation_or_hazard"),
            *[match("draw") for _ in range(5)],
        ]
        report = summarize_matches(
            matches,
            baseline_sha256="a" * 64,
            candidate_sha256="b" * 64,
            seeds=seeds,
        )
        self.assertEqual(report["candidate"]["loss_profile"]["early_elimination"], 2)
        self.assertEqual(report["candidate"]["loss_profile"]["starvation_or_hazard"], 1)

    def test_training_seed_cannot_leak_into_held_out_set(self):
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            validate_held_out_seeds([2026, 7103], training_seed=2026)


if __name__ == "__main__":
    unittest.main()
