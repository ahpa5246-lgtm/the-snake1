"""Constant-memory JSON export of completed Battlesnake game replays.

The Cloudflare Worker consumes exactly the same {"games": [...], "activity": ...}
schema as before. Sending bounded chunks avoids materializing hundreds of
full-resolution game traces in a single large list/JSON byte string.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from live_recorder import last_replay_event, replay_activity, replay_dir

CHUNK_BYTES = 64 * 1024


def replay_export_chunks(limit: int = 250) -> Iterator[bytes]:
    limit = max(1, min(500, limit))
    paths = sorted(
        replay_dir().glob("*.jsonl"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0,
        reverse=True,
    )
    activity = replay_activity()
    header = (
        '{"generated_at_utc":'
        + json.dumps(datetime.now(timezone.utc).isoformat())
        + ',"games":['
    ).encode("utf-8")
    yield header
    emitted = 0
    buffer = bytearray()
    for path in paths:
        if emitted >= limit:
            break
        if last_replay_event(path).get("type") != "end":
            continue
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError:
            continue
        with handle:
            # We generate valid JSON event-by-event without retaining all of
            # the file contents. Files are owned by this recorder and the
            # last event marks immutable completion.
            start = (b"," if emitted else b"") + (
                '{"name":' + json.dumps(path.name) + ',"events":['
            ).encode("utf-8")
            buffer.extend(start)
            first = True
            for line in handle:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    # A torn intermediate record should not poison the
                    # entire replay export.
                    continue
                encoded = json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                if not first:
                    buffer.extend(b",")
                buffer.extend(encoded)
                first = False
                if len(buffer) >= CHUNK_BYTES:
                    yield bytes(buffer)
                    buffer.clear()
            buffer.extend(b"]}")
        emitted += 1
        if len(buffer) >= CHUNK_BYTES:
            yield bytes(buffer)
            buffer.clear()
    if buffer:
        yield bytes(buffer)
    yield (
        '],"export_limit":' + str(limit) + ',"activity":'
        + json.dumps(activity, separators=(",", ":"), ensure_ascii=False)
        + "}"
    ).encode("utf-8")
