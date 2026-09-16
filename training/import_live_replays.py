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
from datetime import datetime, timezone
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


def _empty_stats() -> dict[str, Any]:
    return {
        "learning_cycles": 0,
        "total_games": 0,
        "wins": 0,
        "losses": 0,
        "unclassified_games": 0,
        "total_turns": 0,
        "recorded_moves": 0,
        "first_game_utc": None,
        "last_game_utc": None,
        "last_import_utc": None,
    }


def _load_ledger(path: Path) -> tuple[list[str], dict[str, Any]]:
    if not path.is_file():
        return [], _empty_stats()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            return [], _empty_stats()
        processed = [str(item) for item in payload.get("processed", []) if isinstance(item, str)]
        stats = _empty_stats()
        stored_stats = payload.get("stats", {})
        if isinstance(stored_stats, dict):
            for key in stats:
                if key in stored_stats:
                    stats[key] = stored_stats[key]
        if processed and not stored_stats:
            # Preserve the known match count when upgrading a pre-telemetry
            # ledger. Outcome and turn details were not stored historically.
            stats["total_games"] = len(processed)
            stats["unclassified_games"] = len(processed)
        return processed, stats
    except (OSError, ValueError, TypeError):
        return [], _empty_stats()


def import_batch(source: Path, destination: Path, ledger_path: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    games = payload.get("games", [])
    if not isinstance(games, list):
        raise ValueError("replay batch must contain a games list")

    processed, stats = _load_ledger(ledger_path)
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

        end = events[-1]
        stats["total_games"] = int(stats["total_games"]) + 1
        if bool(end.get("won")):
            stats["wins"] = int(stats["wins"]) + 1
        else:
            stats["losses"] = int(stats["losses"]) + 1
        stats["total_turns"] = int(stats["total_turns"]) + max(0, int(end.get("turn", 0) or 0))
        stats["recorded_moves"] = int(stats["recorded_moves"]) + sum(
            1 for event in events if isinstance(event, dict) and event.get("type") == "move"
        )
        ended_at = end.get("ended_at_utc")
        if isinstance(ended_at, str) and ended_at:
            current_first = stats.get("first_game_utc")
            current_last = stats.get("last_game_utc")
            stats["first_game_utc"] = min(current_first, ended_at) if current_first else ended_at
            stats["last_game_utc"] = max(current_last, ended_at) if current_last else ended_at

    processed = processed[-MAX_LEDGER_ENTRIES:]
    stats["learning_cycles"] = int(stats["learning_cycles"]) + 1
    stats["last_import_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(ledger_path, {"schema_version": SCHEMA_VERSION, "processed": processed, "stats": stats})
    cumulative_games = int(stats["total_games"])
    wins = int(stats["wins"])
    return {
        "available_games": len(games),
        "new_games": imported,
        "duplicate_games": duplicates,
        "rejected_games": rejected,
        "ledger_entries": len(processed),
        "cumulative_games": cumulative_games,
        "cumulative_wins": wins,
        "cumulative_losses": int(stats["losses"]),
        "cumulative_unclassified_games": int(stats["unclassified_games"]),
        "cumulative_win_rate": round(wins / cumulative_games, 4) if cumulative_games else 0.0,
        "cumulative_turns": int(stats["total_turns"]),
        "cumulative_recorded_moves": int(stats["recorded_moves"]),
        "learning_cycles": int(stats["learning_cycles"]),
        "first_game_utc": stats.get("first_game_utc"),
        "last_game_utc": stats.get("last_game_utc"),
        "last_import_utc": stats.get("last_import_utc"),
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
