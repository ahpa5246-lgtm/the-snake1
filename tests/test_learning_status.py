from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.build_learning_status import build_status


class LearningStatusTests(unittest.TestCase):
    def test_builds_human_readable_cumulative_status_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "ledger.json"
            imported = root / "import.json"
            promotion = root / "promotion.json"
            ledger.write_text(json.dumps({"stats": {
                "total_games": 8,
                "wins": 5,
                "losses": 3,
                "total_turns": 400,
                "recorded_moves": 390,
                "learning_cycles": 4,
                "first_game_utc": "2026-09-10T00:00:00+00:00",
                "last_game_utc": "2026-09-11T12:00:00+00:00",
            }}), encoding="utf-8")
            imported.write_text('{"new_games": 2}', encoding="utf-8")
            promotion.write_text('{"promoted": false, "win_rate": 0.4, "minimum_win_rate": 0.55}', encoding="utf-8")
            status = build_status(ledger, imported, promotion, root / "missing.pt", "42")
            self.assertEqual(status["matches_processed"], 8)
            self.assertEqual(status["wins"], 5)
            self.assertEqual(status["observed_match_span_hours"], 36.0)
            self.assertEqual(status["last_cycle_new_games"], 2)
            self.assertIsNone(status["ppo_update"])


if __name__ == "__main__":
    unittest.main()
