"""PPO self-play trainer for the Battlesnake policy/value network.

This trainer keeps *real* game resolution inside ``hisss.BattleSnakeGame``;
all rewards are derived from consecutive engine states.  It does not claim a
CUDA step-rate because that depends on the host CPU, GPU, hisss version and
board configuration.  The trainer selects CUDA automatically when available.

Example:
    python3 training/neural_selfplay.py --updates 200 --games-per-update 64
    python3 training/neural_selfplay.py --updates 1000 --workers 8 --resume

A production checkpoint is written only after an explicit benchmark/promotion
step.  Set BATTLESNAKE_MODEL_PATH to the promoted checkpoint for live inference.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from neural_policy import DIRECTIONS, PolicyValueNet, encode_state, masked_distribution, save_checkpoint, torch_required

CHECKPOINT_DIR = REPO_ROOT / "training" / "checkpoints" / "neural"
POOL_PATH = CHECKPOINT_DIR / "opponent_pool.json"
LATEST_PATH = CHECKPOINT_DIR / "latest.pt"
CHAMPION_PATH = CHECKPOINT_DIR / "champion.pt"


@dataclass
class PPOConfig:
    games_per_update: int = 32
    max_turns: int = 400
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.15
    value_coef: float = 0.5
    entropy_coef: float = 0.012
    learning_rate: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 512
    max_grad_norm: float = 0.8
    board_size: int = 25
    snapshot_interval: int = 20
    seed: int = 2026


@dataclass
class PoolEntry:
    name: str
    kind: str  # neural | tactical | random
    path: str | None = None
    games: int = 0
    wins: int = 0
    nash_prior: float = 0.0

    @property
    def win_rate(self) -> float:
        return (self.wins + 1.0) / (self.games + 2.0)


@dataclass
class PrioritizedOpponentPool:
    entries: list[PoolEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "PrioritizedOpponentPool":
        if not path.is_file():
            return cls(entries=[PoolEntry("tactical", "tactical"), PoolEntry("random", "random")])
        try:
            return cls(entries=[PoolEntry(**entry) for entry in json.loads(path.read_text()).get("entries", [])])
        except (OSError, ValueError, TypeError):
            return cls(entries=[PoolEntry("tactical", "tactical"), PoolEntry("random", "random")])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"entries": [asdict(entry) for entry in self.entries]}, indent=2))

    def sample(self, count: int, rng: random.Random) -> list[PoolEntry]:
        """PFSP-balanced sampling plus a Nash prior and uniform floor.

        ``p(1-p)`` emphasizes opponents with win rate near 50%, where the
        gradient is informative; the floor preserves old tactics and avoids
        catastrophic forgetting.  A persistent ``nash_prior`` may be filled
        from a payoff-matrix evaluation when historic snapshots are available.
        """
        if not self.entries:
            return [PoolEntry("random", "random") for _ in range(count)]
        weights = []
        for entry in self.entries:
            p = entry.win_rate
            balanced = max(0.05, (p * (1.0 - p)) ** 0.5)
            weights.append(0.15 + balanced + max(0.0, entry.nash_prior))
        return rng.choices(self.entries, weights=weights, k=count)

    def record(self, names: list[str], trainee_won: bool) -> None:
        selected = set(names)
        for entry in self.entries:
            if entry.name in selected:
                entry.games += 1
                entry.wins += int(trainee_won)

    def add_snapshot(self, path: Path, update: int) -> None:
        name = f"snapshot_{update:06d}"
        self.entries = [entry for entry in self.entries if entry.name != name]
        self.entries.append(PoolEntry(name=name, kind="neural", path=str(path.relative_to(REPO_ROOT))))
        # A bounded historical league is enough for a small local project and
        # keeps re-evaluation cost controlled.
        historical = [entry for entry in self.entries if entry.kind == "neural"]
        if len(historical) > 16:
            to_drop = sorted(historical, key=lambda entry: entry.name)[:-16]
            drop_names = {entry.name for entry in to_drop}
            self.entries = [entry for entry in self.entries if entry.name not in drop_names]


def solve_zero_sum_nash(payoff: list[list[float]], iterations: int = 2000) -> tuple[list[float], list[float]]:
    """Small multiplicative-weights solver for a zero-sum snapshot league.

    ``payoff[i][j]`` is the row player's result against column player.  The
    average mixed strategies provide a stable league prior without pretending
    that an incomplete, sampled game matrix is an exact global equilibrium.
    """
    if not payoff or not payoff[0]:
        return [], []
    rows, cols = len(payoff), len(payoff[0])
    row_log_w, col_log_w = [0.0] * rows, [0.0] * cols
    row_sum, col_sum = [0.0] * rows, [0.0] * cols
    eta = math.sqrt(2.0 * math.log(max(rows, cols) + 1) / max(1, iterations))
    for _ in range(iterations):
        row_max, col_max = max(row_log_w), max(col_log_w)
        row_probs = [math.exp(value - row_max) for value in row_log_w]
        col_probs = [math.exp(value - col_max) for value in col_log_w]
        row_total, col_total = sum(row_probs), sum(col_probs)
        row_probs, col_probs = [value / row_total for value in row_probs], [value / col_total for value in col_probs]
        for index, value in enumerate(row_probs):
            row_sum[index] += value
        for index, value in enumerate(col_probs):
            col_sum[index] += value
        row_gain = [sum(payoff[row][col] * col_probs[col] for col in range(cols)) for row in range(rows)]
        col_loss = [-sum(row_probs[row] * payoff[row][col] for row in range(rows)) for col in range(cols)]
        row_log_w = [weight + eta * gain for weight, gain in zip(row_log_w, row_gain)]
        col_log_w = [weight + eta * loss for weight, loss in zip(col_log_w, col_loss)]
    return [value / iterations for value in row_sum], [value / iterations for value in col_sum]


def _dependencies() -> tuple[Any, Any]:
    """Import heavy optional dependencies only when the training command runs."""
    torch_required()
    import torch
    try:
        import hisss
    except ImportError as exc:
        raise RuntimeError("Training requires hisss; install requirements-training.txt.") from exc
    return torch, hisss


def _direction_map(hisss: Any) -> dict[str, Any]:
    return {"up": hisss.UP, "down": hisss.DOWN, "left": hisss.LEFT, "right": hisss.RIGHT}


def _action_mask(raw: dict[str, Any], game_tag: str) -> set[str]:
    """Mask known immediate deaths, retaining strategically risky legal moves."""
    from main import TacticalEngine, _new_mem_entry, _update_food_memory
    from main import _game_memory  # imported locally to avoid server coupling at module import

    _game_memory.setdefault(game_tag, _new_mem_entry())
    context = TacticalEngine._build_context(raw, _update_food_memory(game_tag, raw), game_tag, float("inf"))
    legal = set()
    for direction, delta in TacticalEngine.DIRECTIONS.items():
        candidate = context.our_head[0] + delta[0], context.our_head[1] + delta[1]
        if not TacticalEngine._is_certain_death(candidate, context):
            legal.add(direction)
    return legal or set(DIRECTIONS)


class NeuralGameAgent:
    def __init__(self, model: Any, device: str, board_size: int, *, stochastic: bool, tag: str) -> None:
        self.model, self.device, self.board_size = model, device, board_size
        self.stochastic, self.tag = stochastic, tag
        self.pending: dict[str, Any] | None = None

    def choose(self, raw: dict[str, Any]) -> str:
        torch, _ = _dependencies()
        legal = _action_mask(raw, self.tag)
        observation = encode_state(raw, self.board_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(observation)
            masked_logits = masked_distribution(logits, legal)
            distribution = torch.distributions.Categorical(logits=masked_logits)
            action = distribution.sample() if self.stochastic else torch.argmax(masked_logits, dim=1)
            index = int(action.item())
            if self.pending is not None:
                raise RuntimeError("Unresolved action record")
            self.pending = {
                "observation": observation.squeeze(0).cpu(), "action": index,
                "log_prob": float(distribution.log_prob(action).item()), "value": float(value.item()),
                "legal": legal,
            }
        return DIRECTIONS[index]


class RandomGameAgent:
    def __init__(self, rng: random.Random, tag: str) -> None:
        self.rng, self.tag = rng, tag

    def choose(self, raw: dict[str, Any]) -> str:
        return self.rng.choice(sorted(_action_mask(raw, self.tag)))


class TacticalGameAgent:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def choose(self, raw: dict[str, Any]) -> str:
        from main import TacticalEngine, _game_memory, _new_mem_entry, _update_enemy_memory, _update_food_memory
        _game_memory.setdefault(self.tag, _new_mem_entry())
        _update_enemy_memory(self.tag, raw)
        return TacticalEngine.get_best_move(raw, _update_food_memory(self.tag, raw), self.tag, time.monotonic() + 0.5)


def _load_snapshot(entry: PoolEntry, device: str) -> Any:
    from neural_policy import load_checkpoint
    if not entry.path:
        raise ValueError("Neural pool entry has no checkpoint path")
    path = Path(entry.path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    model, _metadata, _extra = load_checkpoint(path, device=device)
    return model


def _build_opponent(entry: PoolEntry, device: str, board_size: int, rng: random.Random, tag: str) -> Any:
    if entry.kind == "tactical":
        return TacticalGameAgent(tag)
    if entry.kind == "random":
        return RandomGameAgent(rng, tag)
    try:
        return NeuralGameAgent(_load_snapshot(entry, device), device, board_size, stochastic=True, tag=tag)
    except Exception:
        # A moved/invalid archived checkpoint should not derail long training.
        return TacticalGameAgent(tag)


def _alive_player(raw: dict[str, Any], player_index: int, alive: set[int]) -> bool:
    return player_index in alive and bool(raw.get("you", {}).get("body"))


def transition_reward(before: dict[str, Any], after: dict[str, Any], *, alive_after: bool, won: bool) -> float:
    """Reward derived only from engine-observable, post-resolution outcomes.

    Death has an exact ``-1.0`` terminal reward.  Food is rewarded most when
    health was genuinely scarce; a small length term supports H2H capacity but
    cannot outweigh a death.  Hazards carry a cost on every surviving entry.
    """
    if not alive_after:
        return -1.0
    before_you, after_you = before.get("you", {}), after.get("you", {})
    reward = 0.01  # survival, deliberately small vs terminal result
    previous_head = before_you.get("head", {})
    previous_food = {(item.get("x"), item.get("y")) for item in before.get("board", {}).get("food", [])}
    # The next head, not health reset alone, identifies food under all rulesets.
    current_head = after_you.get("head", {})
    if (current_head.get("x"), current_head.get("y")) in previous_food:
        reward += 0.16 if int(before_you.get("health", 100)) <= 35 else 0.05
    if int(after_you.get("length", 0)) > int(before_you.get("length", 0)):
        reward += 0.025
    current_position = (current_head.get("x"), current_head.get("y"))
    hazard_cells = {(item.get("x"), item.get("y")) for item in after.get("board", {}).get("hazards", [])}
    if current_position in hazard_cells:
        reward -= 0.035
    if won:
        reward += 1.0
    return reward


def _game_config(hisss: Any, players: int) -> Any:
    config = hisss.restricted_standard_config()
    config.num_players = players
    config.all_actions_legal = True
    return config


def collect_game(model: Any, pool: PrioritizedOpponentPool, config: PPOConfig, device: str, rng: random.Random, game_number: int) -> tuple[list[dict[str, Any]], bool, list[str], int]:
    """Run one real-engine FFA and return only the trainee's PPO trajectory."""
    torch, hisss = _dependencies()
    direction_map = _direction_map(hisss)
    selections = pool.sample(3, rng)
    trainee = NeuralGameAgent(model, device, config.board_size, stochastic=True, tag=f"train-{game_number}")
    opponents = [_build_opponent(entry, device, config.board_size, rng, f"opp-{game_number}-{index}") for index, entry in enumerate(selections)]
    players = [trainee, *opponents]
    environment = hisss.BattleSnakeGame(_game_config(hisss, len(players)))
    trajectory: list[dict[str, Any]] = []

    for _turn in range(config.max_turns):
        if environment.is_terminal() or 0 not in set(environment.players_alive()):
            break
        order = list(environment.players_at_turn())
        before_raw: dict[int, dict[str, Any]] = {}
        actions = []
        trainee_before: dict[str, Any] | None = None
        for player_index in order:
            raw = json.loads(hisss.to_battlesnake_json(environment, player_index))
            before_raw[player_index] = raw
            direction = players[player_index].choose(raw)
            if player_index == 0:
                trainee_before = raw
            actions.append(direction_map.get(direction, hisss.UP))
        environment.step(actions=tuple(actions))
        alive = set(environment.players_alive())
        after_raw = json.loads(hisss.to_battlesnake_json(environment, 0, include_eliminated=True))
        if trainee.pending is not None and trainee_before is not None:
            won = environment.is_terminal() and alive == {0}
            record = trainee.pending
            record["reward"] = transition_reward(trainee_before, after_raw, alive_after=0 in alive, won=won)
            record["done"] = 0 not in alive or environment.is_terminal()
            trajectory.append(record)
            trainee.pending = None
    won = set(environment.players_alive()) == {0}
    return trajectory, won, [entry.name for entry in selections], len(trajectory)


def _worker_collect(payload: tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str, int]) -> tuple[list[dict[str, Any]], bool, list[str], int]:
    """CPU worker entry point.  Workers receive immutable model weights and
    their own hisss instance, so no game state or random generator is shared."""
    model_state, config_data, pool_data, seed, game_number = payload
    torch, _ = _dependencies()
    worker_model = PolicyValueNet().cpu()
    worker_model.load_state_dict(model_state, strict=True)
    worker_model.eval()
    configuration = PPOConfig(**config_data)
    worker_pool = PrioritizedOpponentPool(entries=[PoolEntry(**entry) for entry in pool_data])
    return collect_game(worker_model, worker_pool, configuration, "cpu", random.Random(f"{seed}:{game_number}"), game_number)


def collect_games_vectorized(
    model: Any,
    pool: PrioritizedOpponentPool,
    config: PPOConfig,
    device: str,
    rng: random.Random,
    update: int,
    workers: int,
) -> list[tuple[list[dict[str, Any]], bool, list[str], int]]:
    """Collect independent real-engine games across processes.

    Neural inference in rollout workers runs on CPU to avoid CUDA context
    contention; the PPO update remains on ``device``.  This maps naturally to
    100+ games per update by increasing ``--games-per-update`` and ``--workers``
    on a host with adequate CPU cores.  It intentionally does not promise a
    fixed steps/second rate because that is hardware-specific.
    """
    games = config.games_per_update
    if workers <= 1:
        return [collect_game(model, pool, config, device, rng, update * 100000 + index) for index in range(games)]
    torch, _ = _dependencies()
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    config_data = asdict(config)
    pool_data = [asdict(entry) for entry in pool.entries]
    payloads = [(state, config_data, pool_data, str(rng.random()), update * 100000 + index) for index in range(games)]
    process_count = min(workers, games)
    context = mp.get_context("spawn")
    with context.Pool(processes=process_count) as executor:
        return executor.map(_worker_collect, payloads)


def _returns_and_advantages(records: list[dict[str, Any]], gamma: float, gae_lambda: float) -> tuple[Any, Any]:
    torch, _ = _dependencies()
    advantages = torch.zeros(len(records), dtype=torch.float32)
    future_advantage = 0.0
    for index in range(len(records) - 1, -1, -1):
        current = records[index]
        next_value = 0.0 if current["done"] or index + 1 == len(records) else float(records[index + 1]["value"])
        nonterminal = 0.0 if current["done"] else 1.0
        delta = float(current["reward"]) + gamma * next_value * nonterminal - float(current["value"])
        future_advantage = delta + gamma * gae_lambda * nonterminal * future_advantage
        advantages[index] = future_advantage
    values = torch.tensor([float(record["value"]) for record in records], dtype=torch.float32)
    return advantages, advantages + values


def ppo_update(model: Any, optimizer: Any, records: list[dict[str, Any]], config: PPOConfig, device: str) -> dict[str, float]:
    torch, _ = _dependencies()
    if not records:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    advantages, returns = _returns_and_advantages(records, config.gamma, config.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    observations = torch.stack([record["observation"] for record in records]).to(device)
    actions = torch.tensor([record["action"] for record in records], dtype=torch.long, device=device)
    old_log_prob = torch.tensor([record["log_prob"] for record in records], dtype=torch.float32, device=device)
    returns, advantages = returns.to(device), advantages.to(device)
    masks = torch.tensor([[direction in record["legal"] for direction in DIRECTIONS] for record in records], dtype=torch.bool, device=device)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "batches": 0}
    indices = list(range(len(records)))
    model.train()
    for _epoch in range(config.epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), config.minibatch_size):
            batch = torch.tensor(indices[start:start + config.minibatch_size], dtype=torch.long, device=device)
            logits, values = model(observations[batch])
            logits = logits.masked_fill(~masks[batch], -1e9)
            distribution = torch.distributions.Categorical(logits=logits)
            log_prob = distribution.log_prob(actions[batch])
            ratio = torch.exp(log_prob - old_log_prob[batch])
            unclipped = ratio * advantages[batch]
            clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages[batch]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (returns[batch] - values).pow(2).mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            totals["policy_loss"] += float(policy_loss.detach().cpu())
            totals["value_loss"] += float(value_loss.detach().cpu())
            totals["entropy"] += float(entropy.detach().cpu())
            totals["batches"] += 1
    model.eval()
    divisor = max(1, totals.pop("batches"))
    return {key: value / divisor for key, value in totals.items()}


def _load_or_create_model(device: str, resume: bool) -> tuple[Any, Any, int, dict[str, Any]]:
    torch, _ = _dependencies()
    if resume and LATEST_PATH.is_file():
        from neural_policy import load_checkpoint
        model, _metadata, extra = load_checkpoint(LATEST_PATH, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(extra.get("learning_rate", 3e-4)))
        if extra.get("optimizer_state"):
            optimizer.load_state_dict(extra["optimizer_state"])
        return model, optimizer, int(extra.get("update", 0)), extra
    model = PolicyValueNet().to(device)
    return model, torch.optim.Adam(model.parameters(), lr=3e-4), 0, {}


def train(args: argparse.Namespace) -> None:
    torch, _ = _dependencies()
    configuration = PPOConfig(games_per_update=args.games_per_update, max_turns=args.max_turns, epochs=args.epochs, seed=args.seed)
    random.seed(configuration.seed)
    torch.manual_seed(configuration.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model, optimizer, start_update, resume_state = _load_or_create_model(device, args.resume)
    pool = PrioritizedOpponentPool.load(POOL_PATH)
    rng = random.Random(configuration.seed)
    if args.resume and resume_state:
        if resume_state.get("python_random_state") is not None:
            random.setstate(resume_state["python_random_state"])
        if resume_state.get("rollout_random_state") is not None:
            rng.setstate(resume_state["rollout_random_state"])
        if resume_state.get("torch_random_state") is not None:
            torch.set_rng_state(resume_state["torch_random_state"].cpu())
        if str(device).startswith("cuda") and resume_state.get("torch_cuda_random_state_all") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in resume_state["torch_cuda_random_state_all"]])
    print(f"device={device} updates={args.updates} games/update={configuration.games_per_update} pool={len(pool.entries)}")
    for update in range(start_update + 1, start_update + args.updates + 1):
        started = time.monotonic()
        records: list[dict[str, Any]] = []
        wins = 0
        games = collect_games_vectorized(model, pool, configuration, device, rng, update, args.workers)
        for trajectory, won, names, _turns in games:
            records.extend(trajectory)
            wins += int(won)
            pool.record(names, won)
        metrics = ppo_update(model, optimizer, records, configuration, device)
        extra = {
            "update": update,
            "seed": configuration.seed,
            "learning_rate": configuration.learning_rate,
            "optimizer_state": optimizer.state_dict(),
            "win_rate": wins / max(1, configuration.games_per_update),
            "python_random_state": random.getstate(),
            "rollout_random_state": rng.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        save_checkpoint(LATEST_PATH, model, extra=extra, board_size=configuration.board_size)
        if update % configuration.snapshot_interval == 0:
            snapshot = CHECKPOINT_DIR / f"snapshot_{update:06d}.pt"
            save_checkpoint(snapshot, model, extra=extra, board_size=configuration.board_size)
            pool.add_snapshot(snapshot, update)
        pool.save(POOL_PATH)
        elapsed = time.monotonic() - started
        print(f"update={update} moves={len(records)} win_rate={wins/configuration.games_per_update:.3f} policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f} entropy={metrics['entropy']:.4f} sec={elapsed:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Battlesnake CNN policy/value model with PPO+PFSP.")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--games-per-update", type=int, default=32)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1, help="Independent hisss rollout processes; use CPU cores, not CUDA contexts.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
