"""Lightweight supervised adaptation from real Battlesnake matches.

Winning moves are retained as demonstrations. In lost games, late/critical
positions are relabelled by the deterministic tactical engine, giving the neural
adviser a safer correction target before PPO self-play resumes.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from neural_policy import DIRECTIONS, PolicyValueNet, encode_state, load_checkpoint, save_checkpoint, torch_required


def teacher_move(state: dict, game_id: str) -> str:
    from main import TacticalEngine, _game_memory, _new_mem_entry, _update_enemy_memory, _update_food_memory
    _game_memory.setdefault(game_id, _new_mem_entry())
    _update_enemy_memory(game_id, state)
    return TacticalEngine.get_best_move(state, _update_food_memory(game_id, state), game_id, time.monotonic() + 1.0)


def load_examples(directory: Path, tail_turns: int) -> list[tuple[dict, str]]:
    examples: list[tuple[dict, str]] = []
    for path in sorted(directory.glob("*.jsonl")):
        try:
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError):
            continue
        if not events or events[-1].get("type") != "end":
            continue
        won = bool(events[-1].get("won"))
        moves = [event for event in events if event.get("type") == "move" and event.get("move") in DIRECTIONS]
        if won:
            selected = moves
        else:
            selected = moves[-tail_turns:]
        for event in selected:
            state = event.get("state")
            if not isinstance(state, dict):
                continue
            action = event["move"] if won else teacher_move(state, f"teacher-{path.stem}-{event.get('turn', 0)}")
            examples.append((state, action))
    random.Random(260915).shuffle(examples)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=Path, default=Path("training/live_games"))
    parser.add_argument("--checkpoint", type=Path, default=Path("training/checkpoints/neural/latest.pt"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--tail-turns", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=3000)
    args = parser.parse_args()
    examples = load_examples(args.replays, args.tail_turns)[-args.max_examples:]
    if not examples:
        print("No completed live games available; skipping live fine-tune.")
        return

    torch_required()
    import torch
    if args.checkpoint.is_file():
        model, meta, extra = load_checkpoint(args.checkpoint, device="cpu")
        board_size = meta.board_size
    else:
        model, extra, board_size = PolicyValueNet(), {}, 25
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(max(1, args.epochs)):
        total = 0.0
        for start in range(0, len(examples), 64):
            batch = examples[start:start + 64]
            x = torch.stack([encode_state(state, board_size) for state, _ in batch])
            y = torch.tensor([DIRECTIONS.index(action) for _, action in batch], dtype=torch.long)
            logits, _value = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        print(f"live_epoch={epoch + 1} examples={len(examples)} loss={total / len(examples):.4f}")
    extra = dict(extra)
    extra["live_examples"] = int(extra.get("live_examples", 0)) + len(examples)
    save_checkpoint(args.checkpoint, model, extra=extra, board_size=board_size)


if __name__ == "__main__":
    main()
