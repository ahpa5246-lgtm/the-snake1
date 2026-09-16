import json
import os
import time

import live_recorder


def test_replay_activity_detects_recent_unfinished_game(tmp_path, monkeypatch):
    monkeypatch.setattr(live_recorder, "_REPLAY_DIR", tmp_path)
    active = tmp_path / "active.jsonl"
    active.write_text(json.dumps({"type": "move", "game_id": "active", "turn": 7}) + "\n")
    finished = tmp_path / "finished.jsonl"
    finished.write_text(json.dumps({"type": "end", "game_id": "finished", "turn": 9}) + "\n")
    old = time.time() - 600
    os.utime(finished, (old, old))

    activity = live_recorder.replay_activity(active_window_seconds=120)

    assert activity["currently_playing"] is True
    assert activity["active_games"] == 1
    assert activity["last_activity_utc"] is not None
