"""Frozen, held-out evaluation for neural Battlesnake checkpoints.

This module deliberately keeps evaluation separate from training.  It compares
an immutable pre-training baseline with the newly trained candidate on fixed,
held-out scenarios, writes a machine-readable report, and never promotes a
checkpoint.  A positive result means only "eligible for manual review".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
PROTOCOL = "frozen-held-out-hisss-v1"
DEFAULT_HELD_OUT_SEEDS = (7103, 7109, 7121, 7127, 7211, 7213, 7219, 7229)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_held_out_seeds(seeds: Iterable[int], training_seed: int) -> list[int]:
    normalized = [int(seed) for seed in seeds]
    if len(normalized) != len(set(normalized)):
        raise ValueError("evaluation seeds must be unique")
    if training_seed in normalized:
        raise ValueError("evaluation seeds must not overlap the training seed")
    if not normalized:
        raise ValueError("at least one held-out evaluation seed is required")
    return normalized


def _rate(numerator: float, denominator: int) -> float:
    return round(numerator / max(1, denominator), 6)


def summarize_matches(
    matches: list[dict[str, Any]],
    *,
    baseline_sha256: str,
    candidate_sha256: str,
    seeds: list[int],
    minimum_games: int = 8,
    minimum_score: float = 0.55,
) -> dict[str, Any]:
    """Build the review gate report without importing torch or hisss."""
    if len(matches) != len(seeds):
        raise ValueError("one match result is required for every held-out seed")

    wins = sum(match["outcome"] == "candidate_win" for match in matches)
    losses = sum(match["outcome"] == "baseline_win" for match in matches)
    draws = len(matches) - wins - losses
    candidate_points = wins + 0.5 * draws
    candidate_illegal = sum(int(match["candidate"]["invalid_directions"]) for match in matches)
    baseline_illegal = sum(int(match["baseline"]["invalid_directions"]) for match in matches)
    failure_counts = Counter(
        str(match["candidate"].get("failure_reason") or "unknown")
        for match in matches
        if match["outcome"] == "baseline_win"
    )

    enough_evidence = len(matches) >= minimum_games
    distinct_candidate = baseline_sha256 != candidate_sha256
    score = _rate(candidate_points, len(matches))
    gate_passed = (
        enough_evidence
        and distinct_candidate
        and candidate_illegal == 0
        and score >= minimum_score
        and wins > losses
    )
    blockers: list[str] = []
    if not enough_evidence:
        blockers.append("insufficient_games")
    if not distinct_candidate:
        blockers.append("candidate_matches_baseline")
    if candidate_illegal:
        blockers.append("invalid_directions_observed")
    if score < minimum_score:
        blockers.append("score_below_threshold")
    if wins <= losses:
        blockers.append("no_positive_win_margin")

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "promotion_performed": False,
        "checkpoints": {
            "baseline_sha256": baseline_sha256,
            "candidate_sha256": candidate_sha256,
            "distinct": distinct_candidate,
        },
        "held_out_seeds": seeds,
        "thresholds": {
            "minimum_games": minimum_games,
            "minimum_score": minimum_score,
            "maximum_invalid_directions": 0,
            "requires_positive_win_margin": True,
        },
        "candidate": {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "score": score,
            "invalid_directions": candidate_illegal,
            "mean_survival_turns": round(fmean(match["candidate"]["survival_turns"] for match in matches), 3),
            "mean_final_length": round(fmean(match["candidate"]["final_length"] for match in matches), 3),
            "loss_profile": dict(sorted(failure_counts.items())),
        },
        "baseline": {
            "wins": losses,
            "losses": wins,
            "draws": draws,
            "score": _rate(losses + 0.5 * draws, len(matches)),
            "invalid_directions": baseline_illegal,
            "mean_survival_turns": round(fmean(match["baseline"]["survival_turns"] for match in matches), 3),
            "mean_final_length": round(fmean(match["baseline"]["final_length"] for match in matches), 3),
        },
        "gate": {
            "passed": gate_passed,
            "recommendation": "eligible_for_manual_review" if gate_passed else "retain_frozen_baseline",
            "blockers": blockers,
        },
        "matches": matches,
    }


def _fixed_scenario(hisss: Any, seed: int) -> Any:
    """Create a scenario whose initial state is completely derived from seed."""
    config = hisss.restricted_standard_config()
    config.all_actions_legal = True
    config.food_spawn_chance = 0
    config.min_food = 0
    config.init_snake_pos = {
        0: [[2, 2], [2, 1], [2, 0]],
        1: [[12, 12], [12, 13], [12, 14]],
        2: [[2, 12], [2, 13], [2, 14]],
        3: [[12, 2], [12, 1], [12, 0]],
    }
    occupied = {tuple(cell) for body in config.init_snake_pos.values() for cell in body}
    cells = [
        [x, y]
        for y in range(1, 14)
        for x in range(1, 14)
        if (x, y) not in occupied
    ]
    random.Random(seed).shuffle(cells)
    config.init_food_pos = cells[:12]
    return config


class _GreedyNeuralAgent:
    def __init__(self, model: Any, device: str, board_size: int, tag: str) -> None:
        self.model = model
        self.device = device
        self.board_size = board_size
        self.tag = tag

    def choose(self, raw: dict[str, Any]) -> str:
        import torch
        from neural_policy import DIRECTIONS, encode_state, masked_distribution
        from training.neural_selfplay import _action_mask

        legal = _action_mask(raw, self.tag)
        observation = encode_state(raw, self.board_size).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits, _value = self.model(observation)
            action = int(torch.argmax(masked_distribution(logits, legal), dim=1).item())
        return DIRECTIONS[action]


def _play_match(
    candidate_model: Any,
    baseline_model: Any,
    *,
    board_size: int,
    seed: int,
    game_number: int,
    max_turns: int,
) -> dict[str, Any]:
    import hisss
    import torch
    from training.neural_selfplay import TacticalGameAgent, _direction_map

    random.seed(seed)
    torch.manual_seed(seed)
    candidate_slot = game_number % 4
    baseline_slot = (candidate_slot + 2) % 4
    agents: list[Any] = [
        TacticalGameAgent(f"eval-{seed}-tactical-{slot}") for slot in range(4)
    ]
    agents[candidate_slot] = _GreedyNeuralAgent(
        candidate_model, "cpu", board_size, f"eval-{seed}-candidate"
    )
    agents[baseline_slot] = _GreedyNeuralAgent(
        baseline_model, "cpu", board_size, f"eval-{seed}-baseline"
    )

    environment = hisss.BattleSnakeGame(_fixed_scenario(hisss, seed))
    direction_map = _direction_map(hisss)
    invalid = [0, 0, 0, 0]
    elimination_turn: list[int | None] = [None, None, None, None]
    previous_alive = set(environment.players_alive())
    turn = 0

    while not environment.is_terminal() and turn < max_turns:
        turn += 1
        order = list(environment.players_at_turn())
        actions = []
        for player_index in order:
            raw = json.loads(hisss.to_battlesnake_json(environment, player_index))
            direction = agents[player_index].choose(raw)
            action = direction_map.get(direction)
            if action is None:
                invalid[player_index] += 1
                action = hisss.UP
            actions.append(action)
        environment.step(actions=tuple(actions))
        alive = set(environment.players_alive())
        for player_index in previous_alive - alive:
            elimination_turn[player_index] = turn
        previous_alive = alive

    alive = set(environment.players_alive())
    winner = next(iter(alive)) if len(alive) == 1 else None
    lengths = environment.player_lengths()
    healths = environment.player_healths()

    def player_result(slot: int) -> dict[str, Any]:
        survived = slot in alive
        survived_turns = turn if survived else int(elimination_turn[slot] or turn)
        if survived:
            failure_reason = None
        elif invalid[slot]:
            failure_reason = "invalid_direction"
        elif int(healths[slot]) <= 0:
            failure_reason = "starvation_or_hazard"
        elif survived_turns < max(20, math.floor(turn / 3)):
            failure_reason = "early_elimination"
        else:
            failure_reason = "late_elimination"
        score = (
            (4.0 if winner == slot else 0.0)
            + (1.0 if survived else 0.0)
            + survived_turns / max(1, turn)
            + min(1.0, int(lengths[slot]) / 20.0)
        )
        return {
            "slot": slot,
            "survived": survived,
            "survival_turns": survived_turns,
            "final_length": int(lengths[slot]),
            "final_health": int(healths[slot]),
            "invalid_directions": invalid[slot],
            "failure_reason": failure_reason,
            "score": round(score, 6),
        }

    candidate = player_result(candidate_slot)
    baseline = player_result(baseline_slot)
    if candidate["score"] > baseline["score"]:
        outcome = "candidate_win"
    elif baseline["score"] > candidate["score"]:
        outcome = "baseline_win"
    else:
        outcome = "draw"
    return {
        "seed": seed,
        "turns": turn,
        "winner_slot": winner,
        "outcome": outcome,
        "candidate": candidate,
        "baseline": baseline,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from neural_policy import load_checkpoint

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    for path in (baseline_path, candidate_path):
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")

    seeds = validate_held_out_seeds(args.seeds, args.training_seed)
    baseline_sha = sha256_file(baseline_path)
    candidate_sha = sha256_file(candidate_path)
    baseline_model, baseline_meta, _ = load_checkpoint(baseline_path, device="cpu")
    candidate_model, candidate_meta, _ = load_checkpoint(candidate_path, device="cpu")
    if baseline_meta != candidate_meta:
        raise ValueError("candidate and baseline checkpoint metadata differ")

    matches = [
        _play_match(
            candidate_model,
            baseline_model,
            board_size=candidate_meta.board_size,
            seed=seed,
            game_number=index,
            max_turns=args.max_turns,
        )
        for index, seed in enumerate(seeds)
    ]
    if sha256_file(baseline_path) != baseline_sha or sha256_file(candidate_path) != candidate_sha:
        raise RuntimeError("evaluation mutated a frozen checkpoint")

    report = summarize_matches(
        matches,
        baseline_sha256=baseline_sha,
        candidate_sha256=candidate_sha,
        seeds=seeds,
        minimum_games=args.minimum_games,
        minimum_score=args.minimum_score,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen neural candidate without promotion.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_HELD_OUT_SEEDS))
    parser.add_argument("--max-turns", type=int, default=250)
    parser.add_argument("--minimum-games", type=int, default=8)
    parser.add_argument("--minimum-score", type=float, default=0.55)
    return parser.parse_args()


if __name__ == "__main__":
    result = evaluate(parse_args())
    print(json.dumps({"gate": result["gate"], "candidate": result["candidate"]}, sort_keys=True))
