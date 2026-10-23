"""Memory regressions for Cloudflare replay export and live tactical state."""
from __future__ import annotations

import asyncio
import json
import os
import time

import live_recorder
import main
import replay_stream
from server import app


def _append_game(directory, name, *, moves=5, completed=True):
    path = directory / (name + ".jsonl")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "start", "game_id": name}) + "\n")
        for turn in range(moves):
            handle.write(json.dumps({
                "type": "move", "turn": turn, "game_id": name,
                "state": {"board": {"snakes": []}, "payload": "m" * 2048},
            }) + "\n")
        if completed:
            handle.write(json.dumps({"type": "end", "game_id": name, "won": True}) + "\n")
    return path


def test_last_event_uses_bounded_tail_read(tmp_path, monkeypatch):
    monkeypatch.setattr(live_recorder, "_REPLAY_DIR", tmp_path)
    path = _append_game(tmp_path, "large", moves=180)
    assert live_recorder.last_replay_event(path)["type"] == "end"
    assert live_recorder.replay_activity()["active_games"] == 0
    _append_game(tmp_path, "active", moves=2, completed=False)
    assert live_recorder.replay_activity()["active_games"] == 1


def test_export_same_schema_and_only_completed_games(tmp_path, monkeypatch):
    monkeypatch.setattr(live_recorder, "_REPLAY_DIR", tmp_path)
    _append_game(tmp_path, "one", moves=6)
    _append_game(tmp_path, "two", moves=4)
    _append_game(tmp_path, "unfinished", moves=2, completed=False)
    chunks = list(replay_stream.replay_export_chunks(limit=10))
    exported = json.loads(b"".join(chunks))
    assert exported["export_limit"] == 10
    assert set(exported) == {"generated_at_utc", "games", "export_limit", "activity"}
    assert {x["name"] for x in exported["games"]} == {"one.jsonl", "two.jsonl"}
    assert len(next(g for g in exported["games"] if g["name"] == "one.jsonl")["events"]) == 8


def test_stream_yields_chunks_before_full_history_is_materialized(tmp_path, monkeypatch):
    monkeypatch.setattr(live_recorder, "_REPLAY_DIR", tmp_path)
    for i in range(18):
        _append_game(tmp_path, f"g{i}", moves=40)
    chunks = list(replay_stream.replay_export_chunks(limit=18))
    assert len(chunks) > 12
    assert max(map(len, chunks)) < 75000
    assert len(json.loads(b"".join(chunks))["games"]) == 18


def test_asgi_cloudflare_export_streams_compatible_json(tmp_path, monkeypatch):
    monkeypatch.setattr(live_recorder, "_REPLAY_DIR", tmp_path)
    _append_game(tmp_path, "finished", moves=3)
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app({
        "type": "http", "http_version": "1.1",
        "method": "GET", "path": "/learning/replays",
        "headers": [], "query_string": b"", "scheme": "http", "server": ("test", 80),
    }, receive, send))
    assert sent[0]["status"] == 200
    assert sent[-1]["more_body"] is False
    assert any(message.get("more_body") for message in sent[1:-1])
    body = b"".join(message.get("body", b"") for message in sent[1:])
    assert json.loads(body)["games"][0]["events"][-1]["type"] == "end"


def test_orphaned_match_memory_is_expired_and_bounded():
    try:
        main._game_memory.clear()
        now = time.monotonic()
        for i in range(400):
            main._game_memory[str(i)] = {**main._new_mem_entry(), "last_seen": now - 4000}
        main._touch_game_memory("current")
        assert set(main._game_memory) == {"current"}
        for i in range(400):
            main._game_memory[f"recent-{i}"] = {
                **main._new_mem_entry(), "last_seen": now,
            }
        main._touch_game_memory("current")
        assert len(main._game_memory) <= main.MAX_REMEMBERED_GAMES
        assert "current" in main._game_memory
    finally:
        main._game_memory.clear()


def test_health_reports_actual_configuration_without_exposing_paths(monkeypatch):
    monkeypatch.setattr(main.NeuralAdvisor, "_status", "not_configured")
    monkeypatch.delenv("BATTLESNAKE_MODEL_PATH", raising=False)
    payload = json.loads(main.on_health().body)
    assert payload["status"] == "ok"
    assert payload["neural_adviser"] == "not_configured"
    assert payload["configured_checkpoint"] is False
    assert isinstance(payload["remembered_games"], int)
    assert "BATTLESNAKE_MODEL_PATH" not in json.dumps(payload)


def test_health_can_report_loaded_checkpoint(monkeypatch):
    monkeypatch.setattr(main.NeuralAdvisor, "_status", "loaded")
    monkeypatch.setenv("BATTLESNAKE_MODEL_PATH", "/secret/path.pt")
    payload = json.loads(main.on_health().body)
    assert payload["neural_adviser"] == "loaded"
    assert payload["configured_checkpoint"] is True
    assert "/secret/path.pt" not in json.dumps(payload)
