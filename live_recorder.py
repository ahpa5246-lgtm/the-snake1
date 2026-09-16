"""Persistent-ish live game recorder for Battlesnake HTTP matches.

Writes one JSONL file per game. Set BATTLESNAKE_REPLAY_DIR to a mounted/persistent
path in production. On ephemeral hosts the files remain useful during the instance
lifetime but are not guaranteed to survive a restart.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPLAY_DIR = Path(os.getenv("BATTLESNAKE_REPLAY_DIR", "training/live_games"))
_LOCK = threading.Lock()


def replay_dir() -> Path:
    _REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    return _REPLAY_DIR


def _game_id(state: dict[str, Any]) -> str:
    raw = str(state.get("game", {}).get("id", "unknown"))
    return "".join(ch for ch in raw if ch.isalnum() or ch in "-_") or "unknown"


def _append(game_id: str, event: dict[str, Any]) -> None:
    path = replay_dir() / f"{game_id}.jsonl"
    line = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def record_start(state: dict[str, Any]) -> None:
    game_id = _game_id(state)
    _append(game_id, {"type": "start", "game_id": game_id, "state": state})


def record_move(state: dict[str, Any], move: str, latency_ms: float) -> None:
    game_id = _game_id(state)
    _append(game_id, {
        "type": "move", "game_id": game_id, "turn": int(state.get("turn", 0)),
        "move": move, "latency_ms": round(float(latency_ms), 3), "state": state,
    })


def record_end(state: dict[str, Any]) -> None:
    game_id = _game_id(state)
    board = state.get("board", {})
    you = state.get("you", {})
    snakes = board.get("snakes", []) or []
    alive_ids = {snake.get("id") for snake in snakes if isinstance(snake, dict)}
    our_id = you.get("id")
    won = bool(our_id and alive_ids == {our_id})
    _append(game_id, {
        "type": "end", "game_id": game_id, "turn": int(state.get("turn", 0)),
        "won": won, "alive": our_id in alive_ids if our_id else False,
        "final_length": int(you.get("length", len(you.get("body", []) or [])) or 0),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": state,
    })
