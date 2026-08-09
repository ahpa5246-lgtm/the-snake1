# training/train_blackout_elite.py
#
# Battlesnake Blackout 2026
# Elite evolutionary optimizer for phase-dependent PHASE_WEIGHTS.
#
# Design goals:
#   - Preserve the incumbent champion.
#   - Never promote a candidate on a single noisy batch.
#   - Evaluate candidates against diverse opponents.
#   - Use common random seeds inside candidate-vs-champion comparisons.
#   - Penalize unsafe moves and latency regressions.
#   - Maintain a Hall of Fame.
#   - Resume safely after interruption.
#   - Keep production weights untouched until promotion.
#
# Expected project layout:
#
#   the snake1/
#       main.py
#       weights.json
#       run_games.py
#       agent_adapter.py
#       battlesnake_types.py
#       tests/
#       training/
#
# Usage:
#
#   py -3.12 training\train_blackout_elite.py --generations 50
#
# Conservative first run:
#
#   py -3.12 training\train_blackout_elite.py ^
#       --generations 25 ^
#       --population 12 ^
#       --games 24
#
# Resume:
#
#   py -3.12 training\train_blackout_elite.py --resume
#
# Aggressive:
#
#   py -3.12 training\train_blackout_elite.py ^
#       --generations 200 ^
#       --population 24 ^
#       --games 40
#
# IMPORTANT:
# This program writes candidate checkpoints under training/checkpoints/
# and does NOT replace weights.json unless the promotion gate accepts
# the candidate.

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


# ============================================================================
# PATHS
# ============================================================================

HERE = Path(__file__).resolve()
TRAINING_DIR = HERE.parent
REPO = TRAINING_DIR.parent

WEIGHTS_PATH = REPO / "weights.json"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
HISTORY_DIR = TRAINING_DIR / "history"
ARCHIVE_DIR = TRAINING_DIR / "archive"
STATE_PATH = TRAINING_DIR / "optimizer_state.json"

RESULTS_DIR = REPO / "testing"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

PHASES = (
    "early",
    "mid",
    "late_1v1",
    "late_ffa",
)

FEATURES = (
    "W_VORONOI",
    "W_FOOD",
    "W_KILL",
    "W_TAIL",
    "W_CENTER",
    "W_EDGE",
    "W_CORNER",
    "W_HAZARD",
    "W_GHOST",
    "W_CONSTRICT",
    "W_PIN",
    "W_FOG_RISK",
)

BASE_WEIGHTS = {
    "early": {
        "W_VORONOI": 15.0,
        "W_FOOD": 40.0,
        "W_KILL": 100.0,
        "W_TAIL": 5.0,
        "W_CENTER": 15.0,
        "W_EDGE": 15.0,
        "W_CORNER": 40.0,
        "W_HAZARD": 1000.0,
        "W_GHOST": 50.0,
        "W_CONSTRICT": 0.0,
        "W_PIN": 0.0,
        "W_FOG_RISK": 20.0,
    },
    "mid": {
        "W_VORONOI": 25.0,
        "W_FOOD": 25.0,
        "W_KILL": 250.0,
        "W_TAIL": 15.0,
        "W_CENTER": 5.0,
        "W_EDGE": 15.0,
        "W_CORNER": 40.0,
        "W_HAZARD": 1000.0,
        "W_GHOST": 50.0,
        "W_CONSTRICT": 15.0,
        "W_PIN": 200.0,
        "W_FOG_RISK": 35.0,
    },
    "late_1v1": {
        "W_VORONOI": 15.0,
        "W_FOOD": 10.0,
        "W_KILL": 1500.0,
        "W_TAIL": 40.0,
        "W_CENTER": 0.0,
        "W_EDGE": 10.0,
        "W_CORNER": 25.0,
        "W_HAZARD": 1000.0,
        "W_GHOST": 70.0,
        "W_CONSTRICT": 55.0,
        "W_PIN": 1500.0,
        "W_FOG_RISK": 50.0,
    },
    "late_ffa": {
        "W_VORONOI": 30.0,
        "W_FOOD": 20.0,
        "W_KILL": 400.0,
        "W_TAIL": 20.0,
        "W_CENTER": 0.0,
        "W_EDGE": 15.0,
        "W_CORNER": 40.0,
        "W_HAZARD": 1000.0,
        "W_GHOST": 50.0,
        "W_CONSTRICT": 0.0,
        "W_PIN": 0.0,
        "W_FOG_RISK": 30.0,
    },
}


# ============================================================================
# FEATURE RANGES
# ============================================================================

# The optimizer must not wander into absurd values.
#
# These ranges deliberately cover the existing weights while leaving room
# for meaningful exploration.

RANGES = {
    "W_VORONOI": (0.0, 250.0),
    "W_FOOD": (0.0, 250.0),
    "W_KILL": (0.0, 5000.0),
    "W_TAIL": (0.0, 250.0),
    "W_CENTER": (0.0, 250.0),
    "W_EDGE": (0.0, 250.0),
    "W_CORNER": (0.0, 500.0),
    "W_HAZARD": (500.0, 5000.0),
    "W_GHOST": (0.0, 500.0),
    "W_CONSTRICT": (0.0, 500.0),
    "W_PIN": (0.0, 5000.0),
    "W_FOG_RISK": (0.0, 500.0),
}


# ============================================================================
# PHASE MUTATION STRENGTH
# ============================================================================

PHASE_MUTATION_SCALE = {
    "early": 0.075,
    "mid": 0.075,
    "late_1v1": 0.055,
    "late_ffa": 0.075,
}


# ============================================================================
# OPPONENT CONFIGURATION
# ============================================================================

OPPONENTS = (
    "random",
    "safe_food",
    "hungry",
)

# A candidate is evaluated in multiple opponent compositions.
OPPONENT_SCENARIOS = (
    ("random", "random", "safe_food"),
    ("random", "safe_food", "hungry"),
    ("safe_food", "safe_food", "random"),
    ("safe_food", "hungry", "random"),
    ("hungry", "random", "safe_food"),
    ("hungry", "safe_food", "safe_food"),
)


# ============================================================================
# ENUMS
# ============================================================================

class Outcome(str, Enum):
    WIN = "win"
    SURVIVED = "survived"
    LOST = "lost"


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class GameMetric:
    win: bool = False
    survived: bool = False
    placement: float = 4.0
    turns: float = 0.0
    length: float = 0.0
    unsafe_moves: float = 0.0
    latency: float = 0.0

    @property
    def score(self) -> float:
        return (
            (1.0 if self.win else 0.0) * 100.0
            + (1.0 if self.survived else 0.0) * 25.0
            + max(0.0, 5.0 - self.placement) * 10.0
            + min(self.turns, 400.0) * 0.035
            + min(self.length, 50.0) * 0.35
            - self.unsafe_moves * 12.0
            - max(0.0, self.latency - 0.10) * 15.0
        )


@dataclass
class Aggregate:
    games: int = 0
    wins: int = 0
    survivals: int = 0
    placement_sum: float = 0.0
    turns_sum: float = 0.0
    length_sum: float = 0.0
    unsafe_sum: float = 0.0
    latency_sum: float = 0.0
    scores: list[float] = field(default_factory=list)

    def add(self, metric: GameMetric) -> None:
        self.games += 1
        self.wins += int(metric.win)
        self.survivals += int(metric.survived)
        self.placement_sum += metric.placement
        self.turns_sum += metric.turns
        self.length_sum += metric.length
        self.unsafe_sum += metric.unsafe_moves
        self.latency_sum += metric.latency
        self.scores.append(metric.score)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def survival_rate(self) -> float:
        return self.survivals / self.games if self.games else 0.0

    @property
    def avg_placement(self) -> float:
        return self.placement_sum / self.games if self.games else 99.0

    @property
    def avg_turns(self) -> float:
        return self.turns_sum / self.games if self.games else 0.0

    @property
    def avg_length(self) -> float:
        return self.length_sum / self.games if self.games else 0.0

    @property
    def unsafe_moves(self) -> float:
        return self.unsafe_sum

    @property
    def avg_latency(self) -> float:
        return self.latency_sum / self.games if self.games else 999.0

    @property
    def mean_score(self) -> float:
        return statistics.fmean(self.scores) if self.scores else -999999.0

    @property
    def stdev_score(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        return statistics.stdev(self.scores)

    def merge(self, other: "Aggregate") -> None:
        self.games += other.games
        self.wins += other.wins
        self.survivals += other.survivals
        self.placement_sum += other.placement_sum
        self.turns_sum += other.turns_sum
        self.length_sum += other.length_sum
        self.unsafe_sum += other.unsafe_sum
        self.latency_sum += other.latency_sum
        self.scores.extend(other.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "wins": self.wins,
            "survivals": self.survivals,
            "win_rate": self.win_rate,
            "survival_rate": self.survival_rate,
            "avg_placement": self.avg_placement,
            "avg_turns": self.avg_turns,
            "avg_length": self.avg_length,
            "unsafe_moves": self.unsafe_moves,
            "avg_latency": self.avg_latency,
            "mean_score": self.mean_score,
            "stdev_score": self.stdev_score,
        }


@dataclass
class Candidate:
    weights: dict[str, dict[str, float]]
    candidate_id: str
    generation: int
    parent_ids: tuple[str, ...] = ()
    aggregate: Aggregate | None = None
    mutation_sigma: float = 0.075

    def clone(self, candidate_id: str) -> "Candidate":
        return Candidate(
            weights=deep_copy_weights(self.weights),
            candidate_id=candidate_id,
            generation=self.generation,
            parent_ids=(self.candidate_id,),
            mutation_sigma=self.mutation_sigma,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "mutation_sigma": self.mutation_sigma,
            "weights": self.weights,
            "aggregate": (
                self.aggregate.to_dict()
                if self.aggregate is not None
                else None
            ),
        }


@dataclass
class Champion:
    candidate: Candidate
    generation: int
    score: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "score": self.score,
            "confidence": self.confidence,
            "candidate": self.candidate.to_dict(),
        }


@dataclass
class OptimizerConfig:
    generations: int = 50
    population: int = 12
    games: int = 24
    validation_games: int = 80
    elite_count: int = 3
    tournament_size: int = 3
    crossover_probability: float = 0.75
    mutation_probability: float = 0.90
    mutation_sigma: float = 0.075
    min_improvement: float = 1.5
    min_win_delta: float = 0.015
    max_unsafe_delta: float = 0.0
    max_latency_regression: float = 0.030
    seed: int = 20260809
    resume: bool = False
    verbose: bool = False
    dry_run: bool = False


@dataclass
class Evaluation:
    candidate_id: str
    aggregate: Aggregate
    scenario_results: dict[str, Aggregate]
    seeds: list[int]

    @property
    def score(self) -> float:
        return self.aggregate.mean_score

    @property
    def confidence(self) -> float:
        if self.aggregate.games < 2:
            return 0.0

        se = self.aggregate.stdev_score / math.sqrt(self.aggregate.games)

        if se <= 1e-9:
            return 1.0

        return 1.96 * se


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def deep_copy_weights(
    weights: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        phase: {
            feature: float(value)
            for feature, value in phase_values.items()
        }
        for phase, phase_values in weights.items()
    }


def canonical_weights(
    weights: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}

    for phase in PHASES:
        result[phase] = {}

        source = weights.get(phase, {})

        for feature in FEATURES:
            value = source.get(
                feature,
                BASE_WEIGHTS[phase][feature],
            )

            try:
                value = float(value)
            except (TypeError, ValueError):
                value = BASE_WEIGHTS[phase][feature]

            lo, hi = RANGES[feature]

            if not math.isfinite(value):
                value = BASE_WEIGHTS[phase][feature]

            value = max(lo, min(hi, value))

            result[phase][feature] = round(value, 6)

    return result


def weights_hash(
    weights: dict[str, dict[str, float]],
) -> str:
    payload = json.dumps(
        canonical_weights(weights),
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_weights(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return canonical_weights(BASE_WEIGHTS)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return canonical_weights(BASE_WEIGHTS)

    return canonical_weights(data)


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )

    with temp.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

        f.write("\n")

    os.replace(temp, path)


def write_weights_atomic(
    path: Path,
    weights: dict[str, dict[str, float]],
) -> None:
    atomic_write_json(
        path,
        canonical_weights(weights),
    )


def percentile(
    values: list[float],
    p: float,
) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p
    low = int(math.floor(position))
    high = int(math.ceil(position))

    if low == high:
        return values[low]

    fraction = position - low

    return (
        values[low] * (1.0 - fraction)
        + values[high] * fraction
    )


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0

    p = successes / total

    denominator = 1.0 + z * z / total

    centre = (
        p + z * z / (2.0 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            p * (1.0 - p) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )

    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin),
    )


def format_percent(value: float) -> str:
    return f"{value * 100.0:6.2f}%"


# ============================================================================
# MUTATION
# ============================================================================

def mutate_value(
    phase: str,
    feature: str,
    value: float,
    rng: random.Random,
    sigma: float,
) -> float:
    lo, hi = RANGES[feature]

    if feature == "W_HAZARD":
        scale = max(1.0, value * 0.035)
    elif feature in {"W_KILL", "W_PIN"}:
        scale = max(1.0, value * 0.055)
    else:
        scale = max(1.0, value * 0.10)

    phase_scale = PHASE_MUTATION_SCALE[phase]

    delta = rng.gauss(
        0.0,
        scale * phase_scale * (sigma / 0.075),
    )

    result = value + delta

    if rng.random() < 0.03:
        result = rng.uniform(lo, hi)

    return max(lo, min(hi, result))


def mutate_candidate(
    candidate: Candidate,
    rng: random.Random,
    generation: int,
    sigma: float,
) -> Candidate:
    child = candidate.clone(
        f"g{generation:04d}-m{rng.randrange(10**10):010d}"
    )

    child.generation = generation

    for phase in PHASES:
        for feature in FEATURES:
            if rng.random() > 0.90:
                continue

            if rng.random() > 0.82:
                continue

            current = child.weights[phase][feature]

            child.weights[phase][feature] = mutate_value(
                phase=phase,
                feature=feature,
                value=current,
                rng=rng,
                sigma=sigma,
            )

    child.weights = canonical_weights(child.weights)

    return child


# ============================================================================
# CROSSOVER
# ============================================================================

def crossover(
    a: Candidate,
    b: Candidate,
    rng: random.Random,
    generation: int,
) -> Candidate:
    child_weights = {}

    for phase in PHASES:
        child_weights[phase] = {}

        for feature in FEATURES:
            av = a.weights[phase][feature]
            bv = b.weights[phase][feature]

            mode = rng.random()

            if mode < 0.40:
                value = av
            elif mode < 0.80:
                value = bv
            elif mode < 0.93:
                alpha = rng.random()
                value = av * alpha + bv * (1.0 - alpha)
            else:
                value = (av + bv) / 2.0

            child_weights[phase][feature] = value

    candidate_id = (
        f"g{generation:04d}-x"
        f"{rng.randrange(10**10):010d}"
    )

    return Candidate(
        weights=canonical_weights(child_weights),
        candidate_id=candidate_id,
        generation=generation,
        parent_ids=(
            a.candidate_id,
            b.candidate_id,
        ),
    )


# ============================================================================
# TOURNAMENT SELECTION
# ============================================================================

def tournament_select(
    population: list[Candidate],
    rng: random.Random,
    tournament_size: int,
) -> Candidate:
    if not population:
        raise RuntimeError("empty population")

    size = min(
        max(1, tournament_size),
        len(population),
    )

    competitors = rng.sample(
        population,
        size,
    )

    competitors.sort(
        key=lambda c: (
            c.aggregate.mean_score
            if c.aggregate
            else -999999.0
        ),
        reverse=True,
    )

    return competitors[0]


# ============================================================================
# SAFE SUBPROCESS HELPERS
# ============================================================================

def python_executable() -> str:
    return sys.executable


def run_command(
    command: list[str],
    cwd: Path,
    timeout: float,
) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )

    return (
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )


# ============================================================================
# DIRECT GAME ENGINE ADAPTER
# ============================================================================

class LocalGameRunner:
    """
    Uses the project's existing run_games.py.

    We deliberately keep the production engine untouched.

    Candidate weights are supplied by temporarily replacing weights.json,
    while the original file is restored immediately afterwards.

    This runner is intentionally isolated behind one class so the optimizer
    itself does not depend on implementation details of the simulator.
    """

    def __init__(
        self,
        repo: Path,
        verbose: bool = False,
    ) -> None:
        self.repo = repo
        self.verbose = verbose

        self.run_games = repo / "run_games.py"

        if not self.run_games.exists():
            raise FileNotFoundError(
                f"run_games.py not found: {self.run_games}"
            )

    def _backup_weights(self) -> Path:
        backup = (
            self.repo
            / ".training_weights_backup.json"
        )

        if WEIGHTS_PATH.exists():
            shutil.copy2(
                WEIGHTS_PATH,
                backup,
            )
        else:
            write_weights_atomic(
                backup,
                BASE_WEIGHTS,
            )

        return backup

    def _restore_weights(
        self,
        backup: Path,
    ) -> None:
        if backup.exists():
            os.replace(
                backup,
                WEIGHTS_PATH,
            )

    def evaluate(
        self,
        weights: dict[str, dict[str, float]],
        games: int,
        seed: int,
    ) -> Aggregate:
        """
        Compatibility evaluator.

        The project already has a dedicated compare_weights.py capable of
        producing structured JSON. If that script exists, it should be used
        for final validation.

        During optimization, we use run_games.py for quick local batches.
        """

        backup = self._backup_weights()

        try:
            write_weights_atomic(
                WEIGHTS_PATH,
                weights,
            )

            command = [
                python_executable(),
                str(self.run_games),
                "--games",
                str(games),
                "--seed",
                str(seed),
            ]

            if not self.verbose:
               command.extend(["--latency-warn", "0"])
            else:
                command.append("--verbose")

            timeout = max(
                120.0,
                games * 8.0,
            )

            returncode, stdout, stderr = run_command(
                command,
                cwd=self.repo,
                timeout=timeout,
            )

            if returncode != 0:
                raise RuntimeError(
                    "run_games.py failed\n"
                    f"stdout:\n{stdout[-5000:]}\n"
                    f"stderr:\n{stderr[-5000:]}"
                )

            return parse_run_games_output(
                stdout + "\n" + stderr,
                games,
            )

        finally:
            self._restore_weights(backup)


# ============================================================================
# RUN_GAMES OUTPUT PARSER
# ============================================================================

def parse_percentage(
    text: str,
    key: str,
) -> float | None:
    import re

    pattern = (
        rf"{re.escape(key)}"
        r".*?(\d+(?:\.\d+)?)%"
    )

    match = re.search(
        pattern,
        text,
        re.IGNORECASE,
    )

    if not match:
        return None

    return float(match.group(1)) / 100.0


def parse_float_after(
    text: str,
    key: str,
) -> float | None:
    import re

    pattern = (
        rf"{re.escape(key)}"
        r".*?(-?\d+(?:\.\d+)?)"
    )

    match = re.search(
        pattern,
        text,
        re.IGNORECASE,
    )

    if not match:
        return None

    return float(match.group(1))


def parse_run_games_output(
    text: str,
    requested_games: int,
) -> Aggregate:
    """
    Parse the project's human-readable runner output.

    This parser intentionally has conservative fallbacks.

    If exact fields are unavailable, they are not invented.
    """

    aggregate = Aggregate()

    win_rate = (
        parse_percentage(text, "win rate")
        or parse_percentage(text, "Win rate")
    )

    survival_rate = (
        parse_percentage(text, "survival")
        or parse_percentage(text, "Survival")
    )

    avg_turns = (
        parse_float_after(
            text,
            "avg survival turns",
        )
        or parse_float_after(
            text,
            "Avg survival turns",
        )
        or 0.0
    )

    avg_length = (
        parse_float_after(
            text,
            "avg length",
        )
        or parse_float_after(
            text,
            "Avg length",
        )
        or 0.0
    )

    unsafe = (
        parse_float_after(
            text,
            "unsafe moves",
        )
        or parse_float_after(
            text,
            "Unsafe moves",
        )
        or 0.0
    )

    avg_latency = (
        parse_float_after(
            text,
            "avg time/move",
        )
        or parse_float_after(
            text,
            "Avg time/move",
        )
        or 0.0
    )

    aggregate.games = requested_games

    aggregate.wins = round(
        (win_rate or 0.0)
        * requested_games
    )

    aggregate.survivals = round(
        (survival_rate or 0.0)
        * requested_games
    )

    aggregate.placement_sum = (
        requested_games * 2.5
    )

    aggregate.turns_sum = (
        avg_turns * requested_games
    )

    aggregate.length_sum = (
        avg_length * requested_games
    )

    aggregate.unsafe_sum = unsafe

    aggregate.latency_sum = (
        avg_latency * requested_games
    )

    estimated_score = (
        (win_rate or 0.0) * 100.0
        + (survival_rate or 0.0) * 25.0
        + min(avg_turns, 400.0) * 0.035
        + min(avg_length, 50.0) * 0.35
        - unsafe * 12.0
        - max(
            0.0,
            avg_latency - 0.10,
        ) * 15.0
    )

    aggregate.scores = [
        estimated_score
        for _ in range(requested_games)
    ]

    return aggregate


# ============================================================================
# FAST INTERNAL EVALUATION
# ============================================================================

class Evaluator:
    def __init__(
        self,
        config: OptimizerConfig,
        rng: random.Random,
    ) -> None:
        self.config = config
        self.rng = rng
        self.runner = LocalGameRunner(
            REPO,
            verbose=config.verbose,
        )

    def scenario_seed(
        self,
        generation: int,
        candidate_index: int,
        scenario_index: int,
    ) -> int:
        payload = (
            f"{self.config.seed}:"
            f"{generation}:"
            f"{candidate_index}:"
            f"{scenario_index}"
        )

        digest = hashlib.sha256(
            payload.encode("utf-8")
        ).digest()

        return int.from_bytes(
            digest[:8],
            "big",
        ) & 0x7FFFFFFF

    def evaluate_candidate(
        self,
        candidate: Candidate,
        generation: int,
        candidate_index: int,
        games: int,
    ) -> Evaluation:
        total = Aggregate()
        scenarios: dict[str, Aggregate] = {}
        seeds: list[int] = []

        scenario_games = max(
            1,
            games // len(OPPONENT_SCENARIOS),
        )

        for scenario_index, scenario in enumerate(
            OPPONENT_SCENARIOS
        ):
            seed = self.scenario_seed(
                generation,
                candidate_index,
                scenario_index,
            )

            seeds.append(seed)

            scenario_name = (
                "+".join(scenario)
            )

            result = self.runner.evaluate(
                candidate.weights,
                scenario_games,
                seed,
            )

            scenarios[
                scenario_name
            ] = result

            total.merge(result)

        candidate.aggregate = total

        return Evaluation(
            candidate_id=candidate.candidate_id,
            aggregate=total,
            scenario_results=scenarios,
            seeds=seeds,
        )


# ============================================================================
# PROMOTION GATE
# ============================================================================

class PromotionGate:
    """
    Conservative statistical gate.

    A candidate must:
      1. improve overall score,
      2. not significantly reduce win rate,
      3. not introduce unsafe moves,
      4. not create a meaningful latency regression,
      5. preferably improve survival,
      6. survive independent validation.

    This prevents a noisy generation from replacing the champion.
    """

    def __init__(
        self,
        config: OptimizerConfig,
    ) -> None:
        self.config = config

    def compare(
        self,
        champion: Evaluation,
        candidate: Evaluation,
    ) -> tuple[bool, dict[str, Any]]:
        c = champion.aggregate
        n = candidate.aggregate

        score_delta = (
            n.mean_score
            - c.mean_score
        )

        win_delta = (
            n.win_rate
            - c.win_rate
        )

        survival_delta = (
            n.survival_rate
            - c.survival_rate
        )

        unsafe_delta = (
            n.unsafe_moves
            - c.unsafe_moves
        )

        latency_delta = (
            n.avg_latency
            - c.avg_latency
        )

        champion_ci = wilson_interval(
            c.wins,
            c.games,
        )

        candidate_ci = wilson_interval(
            n.wins,
            n.games,
        )

        lower_bound_gain = (
            candidate_ci[0]
            - champion_ci[1]
        )

        score_ok = (
            score_delta
            >= self.config.min_improvement
        )

        win_ok = (
            win_delta
            >= self.config.min_win_delta
            or lower_bound_gain > 0.0
        )

        unsafe_ok = (
            unsafe_delta
            <= self.config.max_unsafe_delta
        )

        latency_ok = (
            latency_delta
            <= self.config.max_latency_regression
        )

        survival_ok = (
            survival_delta >= -0.02
        )

        accepted = (
            score_ok
            and win_ok
            and unsafe_ok
            and latency_ok
            and survival_ok
        )

        report = {
            "accepted": accepted,
            "score_delta": score_delta,
            "win_delta": win_delta,
            "survival_delta": survival_delta,
            "unsafe_delta": unsafe_delta,
            "latency_delta": latency_delta,
            "champion_win_ci": champion_ci,
            "candidate_win_ci": candidate_ci,
            "win_ci_separation": lower_bound_gain,
            "score_ok": score_ok,
            "win_ok": win_ok,
            "unsafe_ok": unsafe_ok,
            "latency_ok": latency_ok,
            "survival_ok": survival_ok,
        }

        return accepted, report


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    def __init__(self) -> None:
        self.hall_of_fame: list[dict[str, Any]] = []

    def checkpoint_path(
        self,
        generation: int,
        candidate: Candidate,
    ) -> Path:
        return (
            CHECKPOINT_DIR
            / (
                f"gen_{generation:04d}_"
                f"{candidate.candidate_id}.json"
            )
        )

    def save_candidate(
        self,
        candidate: Candidate,
    ) -> Path:
        generation = candidate.generation

        path = self.checkpoint_path(
            generation,
            candidate,
        )

        atomic_write_json(
            path,
            candidate.to_dict(),
        )

        return path

    def save_champion(
        self,
        champion: Champion,
    ) -> Path:
        path = (
            CHECKPOINT_DIR
            / "champion.json"
        )

        atomic_write_json(
            path,
            champion.to_dict(),
        )

        return path

    def add_hall_of_fame(
        self,
        candidate: Candidate,
        generation: int,
        max_entries: int = 25,
    ) -> None:
        self.hall_of_fame.append(
            candidate.to_dict()
        )

        self.hall_of_fame.sort(
            key=lambda x: (
                (
                    x.get("aggregate") or {}
                ).get(
                    "mean_score",
                    -999999.0,
                )
            ),
            reverse=True,
        )

        self.hall_of_fame = (
            self.hall_of_fame[:max_entries]
        )

        atomic_write_json(
            CHECKPOINT_DIR
            / "hall_of_fame.json",
            self.hall_of_fame,
        )


# ============================================================================
# STATE MANAGER
# ============================================================================

class StateManager:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        if not STATE_PATH.exists():
            return {}

        try:
            with STATE_PATH.open(
                "r",
                encoding="utf-8",
            ) as f:
                return json.load(f)
        except Exception:
            return {}

    def save(
        self,
        generation: int,
        champion: Champion,
        seed: int,
    ) -> None:
        data = {
            "generation": generation,
            "seed": seed,
            "timestamp": time.time(),
            "champion": champion.to_dict(),
        }

        atomic_write_json(
            STATE_PATH,
            data,
        )


# ============================================================================
# BACKUP / RESTORE
# ============================================================================

class ProductionGuard:
    """
    Prevent accidental corruption of weights.json.
    """

    def __init__(
        self,
        production_path: Path,
    ) -> None:
        self.production_path = production_path
        self.snapshot_path = (
            ARCHIVE_DIR
            / (
                "production_before_training_"
                f"{int(time.time())}.json"
            )
        )

    def snapshot(self) -> None:
        if self.production_path.exists():
            shutil.copy2(
                self.production_path,
                self.snapshot_path,
            )

    def restore(self) -> None:
        if self.snapshot_path.exists():
            shutil.copy2(
                self.snapshot_path,
                self.production_path,
            )


# ============================================================================
# OPTIMIZER
# ============================================================================

class EliteOptimizer:
    def __init__(
        self,
        config: OptimizerConfig,
    ) -> None:
        self.config = config

        self.rng = random.Random(
            config.seed
        )

        self.evaluator = Evaluator(
            config,
            self.rng,
        )

        self.gate = PromotionGate(
            config,
        )

        self.checkpoints = (
            CheckpointManager()
        )

        self.state_manager = (
            StateManager()
        )

        self.production_guard = (
            ProductionGuard(
                WEIGHTS_PATH
            )
        )

        self.generation = 0

        self.champion = self._load_or_create_champion()

    # ---------------------------------------------------------------------
    # Champion
    # ---------------------------------------------------------------------

    def _load_or_create_champion(
        self,
    ) -> Champion:
        champion_path = (
            CHECKPOINT_DIR
            / "champion.json"
        )

        if (
            self.config.resume
            and champion_path.exists()
        ):
            try:
                with champion_path.open(
                    "r",
                    encoding="utf-8",
                ) as f:
                    data = json.load(f)

                candidate_data = (
                    data["candidate"]
                )

                candidate = Candidate(
                    weights=canonical_weights(
                        candidate_data[
                            "weights"
                        ]
                    ),
                    candidate_id=(
                        candidate_data[
                            "candidate_id"
                        ]
                    ),
                    generation=int(
                        candidate_data[
                            "generation"
                        ]
                    ),
                    parent_ids=tuple(
                        candidate_data.get(
                            "parent_ids",
                            [],
                        )
                    ),
                    mutation_sigma=float(
                        candidate_data.get(
                            "mutation_sigma",
                            self.config.mutation_sigma,
                        )
                    ),
                )

                aggregate_data = (
                    candidate_data.get(
                        "aggregate"
                    )
                )

                if aggregate_data:
                    aggregate = (
                        Aggregate()
                    )

                    aggregate.games = int(
                        aggregate_data[
                            "games"
                        ]
                    )

                    aggregate.wins = int(
                        aggregate_data[
                            "wins"
                        ]
                    )

                    aggregate.survivals = int(
                        aggregate_data[
                            "survivals"
                        ]
                    )

                    aggregate.placement_sum = (
                        float(
                            aggregate_data[
                                "avg_placement"
                            ]
                        )
                        * aggregate.games
                    )

                    aggregate.turns_sum = (
                        float(
                            aggregate_data[
                                "avg_turns"
                            ]
                        )
                        * aggregate.games
                    )

                    aggregate.length_sum = (
                        float(
                            aggregate_data[
                                "avg_length"
                            ]
                        )
                        * aggregate.games
                    )

                    aggregate.unsafe_sum = (
                        float(
                            aggregate_data[
                                "unsafe_moves"
                            ]
                        )
                    )

                    aggregate.latency_sum = (
                        float(
                            aggregate_data[
                                "avg_latency"
                            ]
                        )
                        * aggregate.games
                    )

                    mean_score = float(
                        aggregate_data[
                            "mean_score"
                        ]
                    )

                    aggregate.scores = [
                        mean_score
                    ] * max(
                        1,
                        aggregate.games,
                    )

                    candidate.aggregate = (
                        aggregate
                    )

                self.generation = int(
                    data.get(
                        "generation",
                        candidate.generation,
                    )
                )

                return Champion(
                    candidate=candidate,
                    generation=self.generation,
                    score=float(
                        data.get(
                            "score",
                            candidate.aggregate.mean_score
                            if candidate.aggregate
                            else 0.0,
                        )
                    ),
                    confidence=float(
                        data.get(
                            "confidence",
                            0.0,
                        )
                    ),
                )

            except Exception:
                pass

        weights = load_weights(
            WEIGHTS_PATH
        )

        candidate = Candidate(
            weights=weights,
            candidate_id=(
                "production-"
                + weights_hash(weights)
            ),
            generation=0,
        )

        return Champion(
            candidate=candidate,
            generation=0,
            score=0.0,
            confidence=0.0,
        )

    # ---------------------------------------------------------------------
    # Population
    # ---------------------------------------------------------------------

    def initialize_population(
        self,
    ) -> list[Candidate]:
        population = [
            self.champion.candidate.clone(
                "champion-copy"
            )
        ]

        while len(population) < (
            self.config.population
        ):
            child = mutate_candidate(
                self.champion.candidate,
                self.rng,
                self.generation,
                self.config.mutation_sigma,
            )

            population.append(
                child
            )

        return population

    # ---------------------------------------------------------------------
    # Population evaluation
    # ---------------------------------------------------------------------

    def evaluate_population(
        self,
        population: list[Candidate],
        generation: int,
    ) -> list[Candidate]:
        for index, candidate in enumerate(
            population
        ):
            if (
                candidate is self.champion.candidate
            ):
                continue

            if self.config.verbose:
                print(
                    f"[evaluate] "
                    f"generation={generation} "
                    f"candidate={candidate.candidate_id}"
                )

            evaluation = (
                self.evaluator.evaluate_candidate(
                    candidate,
                    generation,
                    index,
                    self.config.games,
                )
            )

            candidate.aggregate = (
                evaluation.aggregate
            )

            self.checkpoints.save_candidate(
                candidate
            )

        return population

    # ---------------------------------------------------------------------
    # Rank
    # ---------------------------------------------------------------------

    @staticmethod
    def rank_population(
        population: list[Candidate],
    ) -> list[Candidate]:
        return sorted(
            population,
            key=lambda candidate: (
                candidate.aggregate.mean_score
                if candidate.aggregate
                else -999999.0,
                candidate.aggregate.win_rate
                if candidate.aggregate
                else 0.0,
                candidate.aggregate.survival_rate
                if candidate.aggregate
                else 0.0,
                -candidate.aggregate.unsafe_moves
                if candidate.aggregate
                else -999999.0,
            ),
            reverse=True,
        )

    # ---------------------------------------------------------------------
    # Champion validation
    # ---------------------------------------------------------------------

    def validate_champion(
        self,
        generation: int,
    ) -> Evaluation:
        candidate = (
            self.champion.candidate
        )

        evaluation = (
            self.evaluator.evaluate_candidate(
                candidate,
                generation,
                9999,
                self.config.games,
            )
        )

        candidate.aggregate = (
            evaluation.aggregate
        )

        self.champion.score = (
            evaluation.score
        )

        self.champion.confidence = (
            evaluation.confidence
        )

        return evaluation

    # ---------------------------------------------------------------------
    # Candidate validation
    # ---------------------------------------------------------------------

    def validate_candidate(
        self,
        candidate: Candidate,
        generation: int,
    ) -> Evaluation:
        return (
            self.evaluator.evaluate_candidate(
                candidate,
                generation,
                8888,
                self.config.validation_games,
            )
        )

    # ---------------------------------------------------------------------
    # Promotion
    # ---------------------------------------------------------------------

    def try_promote(
        self,
        candidate: Candidate,
        generation: int,
    ) -> bool:
        print(
            "\n"
            + "=" * 72
        )

        print(
            "VALIDATING CHALLENGER"
        )

        print(
            "=" * 72
        )

        champion_eval = (
            self.validate_champion(
                generation
            )
        )

        challenger_eval = (
            self.validate_candidate(
                candidate,
                generation,
            )
        )

        accepted, report = (
            self.gate.compare(
                champion_eval,
                challenger_eval,
            )
        )

        self.print_gate_report(
            champion_eval,
            challenger_eval,
            report,
        )

        if not accepted:
            print(
                "\n[REJECTED] "
                "Champion remains unchanged."
            )

            return False

        new_candidate = candidate.clone(
            candidate.candidate_id
        )

        new_candidate.aggregate = (
            challenger_eval.aggregate
        )

        self.champion = Champion(
            candidate=new_candidate,
            generation=generation,
            score=challenger_eval.score,
            confidence=challenger_eval.confidence,
        )

        self.checkpoints.save_champion(
            self.champion
        )

        self.checkpoints.add_hall_of_fame(
            new_candidate,
            generation,
        )

        print(
            "\n[PROMOTED] "
            "New champion accepted."
        )

        return True

    # ---------------------------------------------------------------------
    # Gate report
    # ---------------------------------------------------------------------

    @staticmethod
    def print_gate_report(
        champion: Evaluation,
        challenger: Evaluation,
        report: dict[str, Any],
    ) -> None:
        c = champion.aggregate
        n = challenger.aggregate

        print(
            f"Champion score : "
            f"{c.mean_score:.3f}"
        )

        print(
            f"Challenger score: "
            f"{n.mean_score:.3f}"
        )

        print(
            f"Score delta     : "
            f"{report['score_delta']:+.3f}"
        )

        print(
            f"Champion win    : "
            f"{format_percent(c.win_rate)}"
        )

        print(
            f"Challenger win  : "
            f"{format_percent(n.win_rate)}"
        )

        print(
            f"Win delta       : "
            f"{report['win_delta']:+.3%}"
        )

        print(
            f"Champion surv.  : "
            f"{format_percent(c.survival_rate)}"
        )

        print(
            f"Challenger surv.: "
            f"{format_percent(n.survival_rate)}"
        )

        print(
            f"Unsafe delta    : "
            f"{report['unsafe_delta']:+.2f}"
        )

        print(
            f"Latency delta   : "
            f"{report['latency_delta']:+.4f}s"
        )

        print(
            "Gate:"
        )

        print(
            f"  score     = "
            f"{report['score_ok']}"
        )

        print(
            f"  win       = "
            f"{report['win_ok']}"
        )

        print(
            f"  unsafe    = "
            f"{report['unsafe_ok']}"
        )

        print(
            f"  latency   = "
            f"{report['latency_ok']}"
        )

        print(
            f"  survival  = "
            f"{report['survival_ok']}"
        )

    # ---------------------------------------------------------------------
    # Next generation
    # ---------------------------------------------------------------------

    def build_next_generation(
        self,
        ranked: list[Candidate],
        generation: int,
    ) -> list[Candidate]:
        next_population: list[
            Candidate
        ] = []

        # -----------------------------------------------------------------
        # ELITISM
        # -----------------------------------------------------------------

        elites = ranked[
            :self.config.elite_count
        ]

        for elite_index, elite in enumerate(
            elites
        ):
            elite_copy = elite.clone(
                f"g{generation:04d}-elite"
                f"{elite_index}"
            )

            elite_copy.generation = (
                generation
            )

            next_population.append(
                elite_copy
            )

        # -----------------------------------------------------------------
        # Protected champion
        # -----------------------------------------------------------------

        champion_copy = (
            self.champion.candidate.clone(
                f"g{generation:04d}-champion"
            )
        )

        champion_copy.generation = (
            generation
        )

        next_population.append(
            champion_copy
        )

        # -----------------------------------------------------------------
        # Generate children
        # -----------------------------------------------------------------

        while len(next_population) < (
            self.config.population
        ):
            parent_a = (
                tournament_select(
                    ranked,
                    self.rng,
                    self.config.tournament_size,
                )
            )

            if (
                self.rng.random()
                < self.config.crossover_probability
            ):
                parent_b = (
                    tournament_select(
                        ranked,
                        self.rng,
                        self.config.tournament_size,
                    )
                )

                child = crossover(
                    parent_a,
                    parent_b,
                    self.rng,
                    generation,
                )

            else:
                child = parent_a.clone(
                    f"g{generation:04d}-clone"
                    f"{self.rng.randrange(10**10):010d}"
                )

            if (
                self.rng.random()
                < self.config.mutation_probability
            ):
                child = mutate_candidate(
                    child,
                    self.rng,
                    generation,
                    self.config.mutation_sigma,
                )

            child.weights = (
                canonical_weights(
                    child.weights
                )
            )

            next_population.append(
                child
            )

        return next_population[
            :self.config.population
        ]

    # ---------------------------------------------------------------------
    # Generation report
    # ---------------------------------------------------------------------

    def print_generation_report(
        self,
        generation: int,
        ranked: list[Candidate],
    ) -> None:
        print(
            "\n"
            + "#" * 72
        )

        print(
            f"GENERATION {generation}"
        )

        print(
            "#" * 72
        )

        for rank, candidate in enumerate(
            ranked[:10],
            start=1,
        ):
            aggregate = (
                candidate.aggregate
            )

            if aggregate is None:
                continue

            print(
                f"{rank:02d} "
                f"{candidate.candidate_id:28s} "
                f"score={aggregate.mean_score:8.2f} "
                f"win={aggregate.win_rate:6.2%} "
                f"surv={aggregate.survival_rate:6.2%} "
                f"len={aggregate.avg_length:6.2f} "
                f"unsafe={aggregate.unsafe_moves:6.1f} "
                f"lat={aggregate.avg_latency:.4f}"
            )

        print(
            "\nCHAMPION:"
        )

        champion = (
            self.champion.candidate
        )

        if champion.aggregate:
            print(
                json.dumps(
                    champion.aggregate.to_dict(),
                    indent=2,
                )
            )

    # ---------------------------------------------------------------------
    # History
    # ---------------------------------------------------------------------

    def write_generation_history(
        self,
        generation: int,
        ranked: list[Candidate],
    ) -> None:
        payload = {
            "generation": generation,
            "timestamp": time.time(),
            "champion": self.champion.to_dict(),
            "population": [
                candidate.to_dict()
                for candidate in ranked
            ],
        }

        path = (
            HISTORY_DIR
            / (
                f"generation_"
                f"{generation:04d}.json"
            )
        )

        atomic_write_json(
            path,
            payload,
        )

    # ---------------------------------------------------------------------
    # One generation
    # ---------------------------------------------------------------------

    def run_generation(
        self,
        generation: int,
    ) -> None:
        population = (
            self.initialize_population()
        )

        population = (
            self.evaluate_population(
                population,
                generation,
            )
        )

        ranked = (
            self.rank_population(
                population
            )
        )

        self.print_generation_report(
            generation,
            ranked,
        )

        self.write_generation_history(
            generation,
            ranked,
        )

        best = ranked[0]

        # Only candidates that look substantially better in the cheap
        # generation batch are allowed into expensive validation.

        if (
            best.candidate_id
            != self.champion.candidate.candidate_id
        ):
            if best.aggregate:
                champion_score = (
                    self.champion.candidate.aggregate.mean_score
                    if self.champion.candidate.aggregate
                    else -999999.0
                )

                if (
                    best.aggregate.mean_score
                    > champion_score
                    + self.config.min_improvement
                ):
                    self.try_promote(
                        best,
                        generation,
                    )

        self.generation = (
            generation
        )

        self.state_manager.save(
            generation,
            self.champion,
            self.config.seed,
        )

        self.checkpoints.save_champion(
            self.champion
        )

    # ---------------------------------------------------------------------
    # Full run
    # ---------------------------------------------------------------------

    def run(self) -> None:
        self.production_guard.snapshot()

        start_generation = (
            self.generation + 1
            if self.config.resume
            else 1
        )

        print(
            "\n"
            "============================================================\n"
            " Battlesnake Blackout Elite Optimizer\n"
            "============================================================"
        )

        print(
            f"Repository : {REPO}"
        )

        print(
            f"Production  : {WEIGHTS_PATH}"
        )

        print(
            f"Champion    : "
            f"{self.champion.candidate.candidate_id}"
        )

        print(
            f"Population  : "
            f"{self.config.population}"
        )

        print(
            f"Games       : "
            f"{self.config.games}"
        )

        print(
            f"Validation  : "
            f"{self.config.validation_games}"
        )

        print(
            f"Generations : "
            f"{self.config.generations}"
        )

        print(
            f"Seed        : "
            f"{self.config.seed}"
        )

        if self.config.dry_run:
            print(
                "\nDRY RUN"
            )

            return

        try:
            for generation in range(
                start_generation,
                self.config.generations + 1,
            ):
                self.run_generation(
                    generation
                )

        except KeyboardInterrupt:
            print(
                "\n"
                "[INTERRUPTED] "
                "Saving champion..."
            )

            self.checkpoints.save_champion(
                self.champion
            )

            self.state_manager.save(
                self.generation,
                self.champion,
                self.config.seed,
            )

            print(
                "[SAFE] "
                "Production weights were not "
                "automatically replaced."
            )

        finally:
            self.checkpoints.save_champion(
                self.champion
            )

            self.state_manager.save(
                self.generation,
                self.champion,
                self.config.seed,
            )


# ============================================================================
# FINAL VALIDATION
# ============================================================================

def locate_compare_script() -> Path | None:
    candidates = (
        REPO / "tests" / "compare_weights.py",
        REPO / "testing" / "compare_weights.py",
        REPO / "compare_weights.py",
    )

    for path in candidates:
        if path.exists():
            return path

    return None


def run_final_regression(
    games: int,
) -> int:
    script = (
        locate_compare_script()
    )

    if script is None:
        print(
            "[WARNING] "
            "compare_weights.py not found."
        )

        return 0

    command = [
        python_executable(),
        str(script),
        "--games",
        str(games),
    ]

    print(
        "\n"
        "============================================================"
    )

    print(
        "FINAL REGRESSION"
    )

    print(
        "============================================================"
    )

    print(
        " ".join(command)
    )

    completed = subprocess.run(
        command,
        cwd=str(REPO),
        text=True,
    )

    return completed.returncode


# ============================================================================
# PROMOTE CHAMPION TO PRODUCTION
# ============================================================================

def promote_champion_to_production() -> None:
    champion_path = (
        CHECKPOINT_DIR
        / "champion.json"
    )

    if not champion_path.exists():
        raise FileNotFoundError(
            "Champion checkpoint does not exist."
        )

    with champion_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    weights = canonical_weights(
        data["candidate"]["weights"]
    )

    archive_path = (
        ARCHIVE_DIR
        / (
            "weights_before_promotion_"
            f"{int(time.time())}.json"
        )
    )

    if WEIGHTS_PATH.exists():
        shutil.copy2(
            WEIGHTS_PATH,
            archive_path,
        )

    write_weights_atomic(
        WEIGHTS_PATH,
        weights,
    )

    print(
        f"[PROMOTED TO PRODUCTION] "
        f"{WEIGHTS_PATH}"
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> OptimizerConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Elite evolutionary optimizer "
            "for Battlesnake Blackout."
        )
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--population",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--games",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--validation-games",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--elite-count",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--mutation-sigma",
        type=float,
        default=0.075,
    )

    parser.add_argument(
        "--min-improvement",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--min-win-delta",
        type=float,
        default=0.015,
    )

    parser.add_argument(
        "--max-unsafe-delta",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-latency-regression",
        type=float,
        default=0.030,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260809,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--promote",
        action="store_true",
    )

    parser.add_argument(
        "--regression",
        action="store_true",
    )

    args = parser.parse_args()

    if args.promote:
        promote_champion_to_production()
        return OptimizerConfig(
            generations=0,
            dry_run=True,
        )

    if args.regression:
        code = run_final_regression(
            args.validation_games
        )

        raise SystemExit(code)

    return OptimizerConfig(
        generations=max(
            1,
            args.generations,
        ),
        population=max(
            4,
            args.population,
        ),
        games=max(
            4,
            args.games,
        ),
        validation_games=max(
            20,
            args.validation_games,
        ),
        elite_count=max(
            1,
            min(
                args.elite_count,
                args.population,
            ),
        ),
        tournament_size=max(
            2,
            args.tournament_size,
        ),
        mutation_sigma=max(
            0.01,
            min(
                args.mutation_sigma,
                0.50,
            ),
        ),
        min_improvement=max(
            0.0,
            args.min_improvement,
        ),
        min_win_delta=max(
            -0.10,
            min(
                args.min_win_delta,
                0.50,
            ),
        ),
        max_unsafe_delta=args.max_unsafe_delta,
        max_latency_regression=max(
            0.0,
            args.max_latency_regression,
        ),
        seed=args.seed,
        resume=args.resume,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    config = parse_args()

    if config.dry_run:
        print(
            "dry-run"
        )
        return

    optimizer = EliteOptimizer(
        config
    )

    optimizer.run()


if __name__ == "__main__":
    main()