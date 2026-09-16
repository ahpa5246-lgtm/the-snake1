"""Build a compact, durable status snapshot for the autonomous learner."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neural_policy import load_checkpoint


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def build_status(
    ledger_path: Path,
    import_summary_path: Path,
    promotion_path: Path,
    checkpoint_path: Path,
    run_id: str,
) -> dict[str, Any]:
    ledger = _read_json(ledger_path)
    stats = ledger.get("stats", {}) if isinstance(ledger.get("stats"), dict) else {}
    imported = _read_json(import_summary_path)
    promotion = _read_json(promotion_path)
    ppo_update = None
    if checkpoint_path.is_file():
        _model, _metadata, extra = load_checkpoint(checkpoint_path, device="cpu")
        ppo_update = extra.get("update")

    first = stats.get("first_game_utc")
    last = stats.get("last_game_utc")
    observed_hours = 0.0
    if isinstance(first, str) and isinstance(last, str):
        try:
            observed_hours = max(0.0, (datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds() / 3600)
        except ValueError:
            pass

    games = int(stats.get("total_games", 0) or 0)
    wins = int(stats.get("wins", 0) or 0)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "github_run_id": str(run_id),
        "tracking_started_utc": first,
        "last_match_utc": last,
        "observed_match_span_hours": round(observed_hours, 2),
        "matches_processed": games,
        "wins": wins,
        "losses": int(stats.get("losses", 0) or 0),
        "unclassified_historical_matches": int(stats.get("unclassified_games", 0) or 0),
        "win_rate": round(wins / games, 4) if games else 0.0,
        "total_turns": int(stats.get("total_turns", 0) or 0),
        "recorded_moves": int(stats.get("recorded_moves", 0) or 0),
        "learning_cycles": int(stats.get("learning_cycles", 0) or 0),
        "last_cycle_new_games": int(imported.get("new_games", 0) or 0),
        "ppo_update": ppo_update,
        "candidate_promoted": bool(promotion.get("promoted", False)),
        "candidate_win_rate": promotion.get("win_rate"),
        "promotion_threshold": promotion.get("minimum_win_rate"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--import-summary", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status = build_status(args.ledger, args.import_summary, args.promotion, args.checkpoint, args.run_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
