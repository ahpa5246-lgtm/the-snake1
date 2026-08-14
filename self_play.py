"""Evaluate the current neural policy against its PFSP opponent league.

For learning, use ``training/neural_selfplay.py`` (or the compatibility route
``training/train_blackout_elite.py --neural``).  This script never mutates the
production checkpoint; it is intended as a quick held-out competitive check.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from neural_policy import load_checkpoint, torch_required
from training.neural_selfplay import LATEST_PATH, PPOConfig, PrioritizedOpponentPool, collect_games_vectorized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the latest Battlesnake neural checkpoint against the PFSP pool.")
    parser.add_argument("--checkpoint", type=Path, default=LATEST_PATH)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_required()
    import torch

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}. Train first with training/neural_selfplay.py.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model, _metadata, extra = load_checkpoint(args.checkpoint, device=device)
    configuration = PPOConfig(games_per_update=max(1, args.games), max_turns=max(1, args.max_turns), seed=args.seed)
    pool = PrioritizedOpponentPool.load(Path("training/checkpoints/neural/opponent_pool.json"))
    results = collect_games_vectorized(model, pool, configuration, device, random.Random(args.seed), int(extra.get("update", 0)), max(1, args.workers))
    wins = sum(int(won) for _trajectory, won, _names, _turns in results)
    turns = [turns for _trajectory, _won, _names, turns in results]
    print(f"checkpoint={args.checkpoint}")
    print(f"games={len(results)} wins={wins} win_rate={wins / max(1, len(results)):.3f}")
    print(f"mean_our_turns={sum(turns) / max(1, len(turns)):.1f} pool_entries={len(pool.entries)}")


if __name__ == "__main__":
    main()
