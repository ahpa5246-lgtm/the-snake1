"""Import completed Battlesnake.com games without training on duplicates.

Render exposes a bounded JSON batch from ``/learning/replays``.  This command
validates that batch, materialises new games as JSONL for the live fine-tuner,
and keeps a compact cursor in the GitHub Actions learning-state cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MAX_LEDGER_ENTRIES = 10_000


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _safe_game_id(value: object) -> str:
    cleaned = "".join(ch for ch in str(value) if ch.isalnum() or ch in "-_")
    return cleaned[:128] or "unknown"


def _load_ledger(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            return []
        return [str(item) for item in payload.get("processed", []) if isinstance(item, str)]
    except (OSError, ValueError, TypeError):
        return []


def import_batch(source: Path, destination: Path, ledger_path: Path) -> dict[str, int]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    games = payload.get("games", [])
    if not isinstance(games, list):
        raise ValueError("replay batch must contain a games list")

    processed = _load_ledger(ledger_path)
    processed_set = set(processed)
    imported = duplicates = rejected = 0
    destination.mkdir(parents=True, exist_ok=True)

    for game in games:
        if not isinstance(game, dict) or not isinstance(game.get("events"), list):
            rejected += 1
            continue
        events = game["events"]
        if not events or not isinstance(events[-1], dict) or events[-1].get("type") != "end":
            rejected += 1
            continue
        encoded = json.dumps(events, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        raw_id = events[-1].get("game_id") or game.get("name") or "unknown"
        game_id = _safe_game_id(raw_id)
        key = f"{game_id}:{digest}"
        if key in processed_set:
            duplicates += 1
            continue

        output = destination / f"{game_id}-{digest[:12]}.jsonl"
        output.write_text(
            "".join(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n" for event in events),
            encoding="utf-8",
        )
        processed.append(key)
        processed_set.add(key)
        imported += 1

    processed = processed[-MAX_LEDGER_ENTRIES:]
    _atomic_json(ledger_path, {"schema_version": SCHEMA_VERSION, "processed": processed})
    return {
        "available_games": len(games),
        "new_games": imported,
        "duplicate_games": duplicates,
        "rejected_games": rejected,
        "ledger_entries": len(processed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("training/live_games"))
    parser.add_argument("--ledger", type=Path, default=Path("training/live-replay-ledger.json"))
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    summary = import_batch(args.source, args.destination, args.ledger)
    if args.summary:
        _atomic_json(args.summary, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
