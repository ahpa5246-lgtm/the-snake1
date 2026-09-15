"""Production ASGI entrypoint with transparent live-match recording.

Besides observing Battlesnake traffic, this wrapper exposes a tiny read-only
learning endpoint. The scheduled GitHub learner downloads completed replay data
before training, so Render's ephemeral filesystem is not the long-term store.
"""
from __future__ import annotations

import json
import time
from typing import Any

from main import app as tactical_app
from live_recorder import record_end, record_move, record_start, replay_dir


class LiveReplayMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and scope.get("method") == "GET" and scope.get("path") == "/learning/replays":
            games: list[dict[str, Any]] = []
            for path in sorted(replay_dir().glob("*.jsonl")):
                try:
                    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    # Export only completed games; never train on half a live match.
                    if events and events[-1].get("type") == "end":
                        games.append({"name": path.name, "events": events})
                except (OSError, ValueError, TypeError):
                    continue
            body = json.dumps({"games": games}, separators=(",", ":")).encode("utf-8")
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]})
            await send({"type": "http.response.body", "body": body})
            return

        if scope.get("type") != "http" or scope.get("method") != "POST" or scope.get("path") not in {"/start", "/move", "/end"}:
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        more = True
        while more:
            message = await receive()
            if message.get("type") != "http.request":
                continue
            chunks.append(message.get("body", b""))
            more = bool(message.get("more_body", False))
        request_body = b"".join(chunks)
        replayed = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal replayed
            if replayed:
                return {"type": "http.request", "body": b"", "more_body": False}
            replayed = True
            return {"type": "http.request", "body": request_body, "more_body": False}

        response_chunks: list[bytes] = []

        async def capture_send(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.body":
                response_chunks.append(message.get("body", b""))
            await send(message)

        started = time.monotonic()
        await self.app(scope, replay_receive, capture_send)
        latency_ms = (time.monotonic() - started) * 1000.0

        try:
            state = json.loads(request_body.decode("utf-8"))
            path = scope["path"]
            if path == "/start":
                record_start(state)
            elif path == "/end":
                record_end(state)
            else:
                response = json.loads(b"".join(response_chunks).decode("utf-8"))
                move = response.get("move")
                if move in {"up", "down", "left", "right"}:
                    record_move(state, move, latency_ms)
        except Exception:
            # Telemetry must never cost a live game.
            pass


app = LiveReplayMiddleware(tactical_app)
