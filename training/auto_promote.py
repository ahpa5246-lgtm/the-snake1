"""Benchmark a candidate against the current champion and promote safely."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from neural_policy import load_checkpoint, torch_required
from training.neural_selfplay import PPOConfig, PoolEntry, PrioritizedOpponentPool, collect_games_vectorized


def evaluate(checkpoint: Path, opponent: Path, games: int, workers: int, seed: int) -> float:
    model, _meta, _extra = load_checkpoint(checkpoint, device="cpu")
    pool = PrioritizedOpponentPool(entries=[PoolEntry(name="incumbent", kind="neural", path=str(opponent))])
    config = PPOConfig(games_per_update=games, max_turns=400, seed=seed)
    results = collect_games_vectorized(model, pool, config, "cpu", random.Random(seed), seed, workers)
    return sum(int(won) for _trajectory, won, _names, _turns in results) / max(1, len(results))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--minimum-win-rate", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=260915)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    torch_required()
    if not args.candidate.is_file():
        raise SystemExit(f"Candidate not found: {args.candidate}")
    args.champion.parent.mkdir(parents=True, exist_ok=True)
    result = {"promoted": False, "bootstrap": False, "win_rate": None, "minimum_win_rate": args.minimum_win_rate}
    if not args.champion.is_file():
        shutil.copy2(args.candidate, args.champion)
        result.update({"promoted": True, "bootstrap": True})
        print("PROMOTED bootstrap candidate: no previous champion existed")
        if args.result_json:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            args.result_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return
    win_rate = evaluate(args.candidate, args.champion, args.games, args.workers, args.seed)
    result["win_rate"] = win_rate
    print(f"candidate_vs_champion games={args.games} win_rate={win_rate:.3f} threshold={args.minimum_win_rate:.3f}")
    if win_rate >= args.minimum_win_rate:
        shutil.copy2(args.candidate, args.champion)
        result["promoted"] = True
        print("PROMOTED candidate -> champion")
    else:
        print("REJECTED candidate; incumbent champion remains live")
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
