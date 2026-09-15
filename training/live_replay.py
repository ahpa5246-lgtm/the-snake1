"""Load and analyze Battlesnake live JSONL replays.

The analyzer deliberately reports observable facts rather than pretending it can
prove strategic causality. It extracts hard positions suitable for later offline
imitation/counterfactual evaluation and produces a compact game report.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DIRECTIONS = ("up", "down", "left", "right")


@dataclass
class LiveGame:
    game_id: str
    moves: list[dict[str, Any]]
    end: dict[str, Any] | None

    @property
    def won(self) -> bool:
        return bool(self.end and self.end.get("won"))


def load_game(path: Path) -> LiveGame:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    moves = [event for event in events if event.get("type") == "move"]
    end = next((event for event in reversed(events) if event.get("type") == "end"), None)
    game_id = str((events[0] if events else {}).get("game_id", path.stem))
    return LiveGame(game_id, moves, end)


def iter_games(directory: Path) -> Iterable[LiveGame]:
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.jsonl")):
        try:
            yield load_game(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue


def hard_positions(game: LiveGame, tail_turns: int = 12) -> list[dict[str, Any]]:
    """Return late losing positions plus low-health/high-latency decisions."""
    selected: dict[int, dict[str, Any]] = {}
    if not game.won:
        for event in game.moves[-tail_turns:]:
            selected[int(event.get("turn", 0))] = event
    for event in game.moves:
        state = event.get("state", {})
        health = int(state.get("you", {}).get("health", 100) or 100)
        if health <= 30 or float(event.get("latency_ms", 0.0)) >= 150.0:
            selected[int(event.get("turn", 0))] = event
    return [selected[key] for key in sorted(selected)]


def export_dataset(directory: Path, output: Path, tail_turns: int = 12) -> dict[str, int]:
    games = list(iter_games(directory))
    rows = []
    for game in games:
        for event in hard_positions(game, tail_turns):
            rows.append({
                "game_id": game.game_id, "won": game.won,
                "turn": event.get("turn"), "state": event.get("state"),
                "action": event.get("move"), "latency_ms": event.get("latency_ms"),
                "source": "battlesnake-live",
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    return {"games": len(games), "positions": len(rows), "wins": sum(game.won for game in games)}


def summarize(directory: Path) -> dict[str, Any]:
    games = list(iter_games(directory))
    latencies = [float(m.get("latency_ms", 0.0)) for g in games for m in g.moves]
    return {
        "games": len(games), "wins": sum(g.won for g in games),
        "losses": sum(not g.won for g in games),
        "win_rate": round(sum(g.won for g in games) / len(games), 4) if games else 0.0,
        "moves": sum(len(g.moves) for g in games),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("training/live_games"))
    parser.add_argument("--export", type=Path)
    parser.add_argument("--tail-turns", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(summarize(args.dir), indent=2))
    if args.export:
        print(json.dumps(export_dataset(args.dir, args.export, args.tail_turns), indent=2))


if __name__ == "__main__":
    main()
