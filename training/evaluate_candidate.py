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
import time
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 3
# Earlier reports used frozen-held-out-hisss-v2-mirrored and measured direct
# neural actions. Those scores are not comparable to the served-policy protocol.
PROTOCOL = "frozen-held-out-hisss-v3-deployed-mirrored"
DEFAULT_HELD_OUT_SEEDS = (7103, 7109, 7121, 7127, 7211, 7213, 7219, 7229)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_model_state(model_state: dict[str, Any]) -> str:
    """Digest policy parameters without checkpoint optimizer/RNG metadata."""
    digest = hashlib.sha256()
    for name in sorted(model_state):
        value = model_state[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(tensor.view(-1).numpy().tobytes())
        elif isinstance(value, (bytes, bytearray, memoryview)):
            digest.update(bytes(value))
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr).encode("utf-8")
            )
        digest.update(b"\0")
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


def mirrored_slot_pairs(game_number: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return a seat assignment and its exact candidate/baseline mirror."""
    candidate_slot = game_number % 4
    baseline_slot = (candidate_slot + 2) % 4
    return ((candidate_slot, baseline_slot), (baseline_slot, candidate_slot))


def select_legal_action(
    selected: Any,
    available_actions: Iterable[Any],
    fallback_order: Iterable[Any],
) -> tuple[Any, bool]:
    """Return a legal deterministic action and whether the policy was invalid."""
    available = tuple(available_actions)
    if not available:
        raise RuntimeError("engine returned no available actions for a player at turn")
    if selected in available:
        return selected, False
    for fallback in fallback_order:
        if fallback in available:
            return fallback, True
    return available[0], True


def validate_mirrored_matches(matches: list[dict[str, Any]], seeds: list[int]) -> None:
    """Require both candidate/baseline seat orders for every held-out seed."""
    if len(matches) != len(seeds) * 2:
        raise ValueError("two mirrored match results are required for every held-out seed")
    expected = set(seeds)
    observed = {int(match["seed"]) for match in matches}
    if observed != expected:
        raise ValueError("mirrored match seeds do not match the held-out set")
    for seed in seeds:
        paired = [match for match in matches if int(match["seed"]) == seed]
        placements = {
            (int(match["candidate"]["slot"]), int(match["baseline"]["slot"]))
            for match in paired
        }
        if len(placements) != 2 or not all((baseline, candidate) in placements for candidate, baseline in placements):
            raise ValueError(f"held-out seed {seed} is missing a mirrored seat assignment")


def summarize_matches(
    matches: list[dict[str, Any]],
    *,
    baseline_model_sha256: str,
    candidate_model_sha256: str,
    baseline_checkpoint_sha256: str,
    candidate_checkpoint_sha256: str,
    seeds: list[int],
    minimum_games: int = 16,
    minimum_score: float = 0.55,
) -> dict[str, Any]:
    """Build the review gate report without importing torch or hisss."""
    validate_mirrored_matches(matches, seeds)

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
    distinct_candidate = baseline_model_sha256 != candidate_model_sha256
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
        "policy": "served_tactics_plus_frozen_neural_adviser",
        "move_budget_ms": 200,
        "promotion_performed": False,
        "checkpoints": {
            "baseline_file_sha256": baseline_checkpoint_sha256,
            "candidate_file_sha256": candidate_checkpoint_sha256,
            "distinct_files": baseline_checkpoint_sha256 != candidate_checkpoint_sha256,
        },
        "model_states": {
            "baseline_sha256": baseline_model_sha256,
            "candidate_sha256": candidate_model_sha256,
            "distinct": distinct_candidate,
        },
        "held_out_seeds": seeds,
        "matches_per_seed": 2,
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
    config.all_actions_legal = False
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


class _DeployedPolicyAgent:
    """Frozen checkpoint through the same safety, scoring and adviser as /move."""
    def __init__(self, model: Any, board_size: int, tag: str) -> None:
        from main import NeuralAdvisor
        self.tag = tag
        self.neural_inferences = 0
        self.moves = 0
        self.total_latency_ms = 0.0
        owner = self

        class FrozenAdvisor(NeuralAdvisor):
            _attempted = True
            _model = model
            _board_size = board_size
            _device = "cpu"

            @classmethod
            def scores(cls, data, legal, deadline):
                result = super().scores(data, legal, deadline)
                if result:
                    owner.neural_inferences += 1
                return result

        # Each player owns its class state. No process-global model swapping.
        self.advisor = FrozenAdvisor

    def choose(self, raw: dict[str, Any]) -> str:
        from main import (COMPUTE_BUDGET_S, TacticalEngine, _game_memory,
                          _new_mem_entry, _update_enemy_memory, _update_food_memory)
        started = time.monotonic()
        _game_memory.setdefault(self.tag, _new_mem_entry())
        _update_enemy_memory(self.tag, raw)
        direction = TacticalEngine.get_best_move(
            raw, _update_food_memory(self.tag, raw), self.tag,
            started + COMPUTE_BUDGET_S, advisor=self.advisor,
        )
        self.moves += 1
        self.total_latency_ms += (time.monotonic() - started) * 1000
        return direction


def _play_match(
    candidate_model: Any,
    baseline_model: Any,
    *,
    board_size: int,
    seed: int,
    candidate_slot: int,
    baseline_slot: int,
    mirror_index: int,
    max_turns: int,
) -> dict[str, Any]:
    import hisss
    import torch
    from training.neural_selfplay import _direction_map

    random.seed(seed)
    torch.manual_seed(seed)
    if candidate_slot == baseline_slot:
        raise ValueError("candidate and baseline must occupy different slots")
    agents: list[Any] = [
        _DeployedPolicyAgent(None, board_size, f"eval-{seed}-{mirror_index}-tactical-{slot}") for slot in range(4)
    ]
    agents[candidate_slot] = _DeployedPolicyAgent(
        candidate_model, board_size, f"eval-{seed}-{mirror_index}-candidate"
    )
    agents[baseline_slot] = _DeployedPolicyAgent(
        baseline_model, board_size, f"eval-{seed}-{mirror_index}-baseline"
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
            action, was_invalid = select_legal_action(
                direction_map.get(direction),
                environment.available_actions(player_index),
                direction_map.values(),
            )
            if was_invalid:
                invalid[player_index] += 1
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
            "neural_inferences": agents[slot].neural_inferences,
            "moves": agents[slot].moves,
            "mean_decision_ms": round(agents[slot].total_latency_ms / max(1, agents[slot].moves), 3),
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
    from main import _game_memory
    for agent in agents:
        _game_memory.pop(agent.tag, None)
    if candidate["score"] > baseline["score"]:
        outcome = "candidate_win"
    elif baseline["score"] > candidate["score"]:
        outcome = "baseline_win"
    else:
        outcome = "draw"
    return {
        "seed": seed,
        "mirror_index": mirror_index,
        "turns": turn,
        "winner_slot": winner,
        "outcome": outcome,
        "candidate": candidate,
        "baseline": baseline,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from neural_policy import load_checkpoint
    import torch
    torch.set_num_threads(1)

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    for path in (baseline_path, candidate_path):
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")

    seeds = validate_held_out_seeds(args.seeds, args.training_seed)
    baseline_checkpoint_sha = sha256_file(baseline_path)
    candidate_checkpoint_sha = sha256_file(candidate_path)
    baseline_model, baseline_meta, _ = load_checkpoint(baseline_path, device="cpu")
    candidate_model, candidate_meta, _ = load_checkpoint(candidate_path, device="cpu")
    if baseline_meta != candidate_meta:
        raise ValueError("candidate and baseline checkpoint metadata differ")
    baseline_model_sha = sha256_model_state(baseline_model.state_dict())
    candidate_model_sha = sha256_model_state(candidate_model.state_dict())

    matches = []
    for index, seed in enumerate(seeds):
        for mirror_index, (candidate_slot, baseline_slot) in enumerate(mirrored_slot_pairs(index)):
            matches.append(
                _play_match(
                    candidate_model,
                    baseline_model,
                    board_size=candidate_meta.board_size,
                    seed=seed,
                    candidate_slot=candidate_slot,
                    baseline_slot=baseline_slot,
                    mirror_index=mirror_index,
                    max_turns=args.max_turns,
                )
            )
    if (
        sha256_file(baseline_path) != baseline_checkpoint_sha
        or sha256_file(candidate_path) != candidate_checkpoint_sha
    ):
        raise RuntimeError("evaluation mutated a frozen checkpoint")

    report = summarize_matches(
        matches,
        baseline_model_sha256=baseline_model_sha,
        candidate_model_sha256=candidate_model_sha,
        baseline_checkpoint_sha256=baseline_checkpoint_sha,
        candidate_checkpoint_sha256=candidate_checkpoint_sha,
        seeds=seeds,
        minimum_games=args.minimum_games,
        minimum_score=args.minimum_score,
    )
    report["adviser_activity"] = {
        label: sum(match[label]["neural_inferences"] for match in matches)
        for label in ("candidate", "baseline")
    }
    if any(count == 0 for count in report["adviser_activity"].values()):
        report["gate"]["passed"] = False
        report["gate"]["recommendation"] = "retain_frozen_baseline"
        report["gate"]["blockers"].append("neural_adviser_not_exercised")
    report["environment"] = {"torch_threads": torch.get_num_threads(), "device": "cpu",
                             "torch_version": torch.__version__}
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
    parser.add_argument("--minimum-games", type=int, default=16)
    parser.add_argument("--minimum-score", type=float, default=0.55)
    return parser.parse_args()


if __name__ == "__main__":
    result = evaluate(parse_args())
    print(json.dumps({"gate": result["gate"], "candidate": result["candidate"]}, sort_keys=True))
