from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.import_live_replays import import_batch


def completed_game(game_id: str, *, won: bool = False, turn: int = 1, ended_at: str = "2026-09-16T12:00:00+00:00") -> dict:
    state = {"game": {"id": game_id}, "turn": 1, "board": {"snakes": []}, "you": {"id": "us"}}
    return {
        "name": f"{game_id}.jsonl",
        "events": [
            {"type": "start", "game_id": game_id, "state": state},
            {"type": "move", "game_id": game_id, "turn": 1, "move": "up", "state": state},
            {"type": "end", "game_id": game_id, "turn": turn, "won": won, "ended_at_utc": ended_at, "state": state},
        ],
    }


class LiveReplayImportTests(unittest.TestCase):
    def test_imports_completed_games_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination, ledger = root / "batch.json", root / "games", root / "ledger.json"
            source.write_text(json.dumps({"games": [completed_game("g-1")]}), encoding="utf-8")
            first = import_batch(source, destination, ledger)
            second = import_batch(source, destination, ledger)
            self.assertEqual(first["new_games"], 1)
            self.assertEqual(second["new_games"], 0)
            self.assertEqual(second["duplicate_games"], 1)
            self.assertEqual(second["cumulative_games"], 1)
            self.assertEqual(second["cumulative_losses"], 1)
            self.assertEqual(second["learning_cycles"], 2)
            replay = next(destination.glob("*.jsonl"))
            self.assertEqual(len(replay.read_text(encoding="utf-8").splitlines()), 3)

    def test_rejects_incomplete_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "batch.json"
            game = completed_game("g-2")
            game["events"].pop()
            source.write_text(json.dumps({"games": [game]}), encoding="utf-8")
            summary = import_batch(source, root / "games", root / "ledger.json")
            self.assertEqual(summary["new_games"], 0)
            self.assertEqual(summary["rejected_games"], 1)

    def test_accumulates_durable_match_stats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination, ledger = root / "batch.json", root / "games", root / "ledger.json"
            source.write_text(json.dumps({"games": [
                completed_game("winner", won=True, turn=76, ended_at="2026-09-16T13:00:00+00:00"),
                completed_game("loss", turn=4, ended_at="2026-09-16T12:00:00+00:00"),
            ]}), encoding="utf-8")
            summary = import_batch(source, destination, ledger)
            self.assertEqual(summary["cumulative_games"], 2)
            self.assertEqual(summary["cumulative_wins"], 1)
            self.assertEqual(summary["cumulative_losses"], 1)
            self.assertEqual(summary["cumulative_turns"], 80)
            self.assertEqual(summary["cumulative_recorded_moves"], 2)
            self.assertEqual(summary["cumulative_win_rate"], 0.5)
            self.assertEqual(summary["first_game_utc"], "2026-09-16T12:00:00+00:00")
            self.assertEqual(summary["last_game_utc"], "2026-09-16T13:00:00+00:00")

    def test_migrates_pre_telemetry_ledger_without_losing_match_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination, ledger = root / "batch.json", root / "games", root / "ledger.json"
            ledger.write_text(json.dumps({"schema_version": 1, "processed": ["old-a:digest", "old-b:digest"]}), encoding="utf-8")
            source.write_text('{"games": []}', encoding="utf-8")
            summary = import_batch(source, destination, ledger)
            self.assertEqual(summary["cumulative_games"], 2)
            self.assertEqual(summary["cumulative_unclassified_games"], 2)


if __name__ == "__main__":
    unittest.main()
