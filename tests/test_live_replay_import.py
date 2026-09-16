from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.import_live_replays import import_batch


def completed_game(game_id: str) -> dict:
    state = {"game": {"id": game_id}, "turn": 1, "board": {"snakes": []}, "you": {"id": "us"}}
    return {
        "name": f"{game_id}.jsonl",
        "events": [
            {"type": "start", "game_id": game_id, "state": state},
            {"type": "move", "game_id": game_id, "turn": 1, "move": "up", "state": state},
            {"type": "end", "game_id": game_id, "turn": 1, "won": False, "state": state},
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


if __name__ == "__main__":
    unittest.main()
