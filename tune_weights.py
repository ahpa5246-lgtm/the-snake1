"""
tune_weights.py — evolutionary self-play tuner for main.PHASE_WEIGHTS.

Does NOT touch engine logic. Only searches for better PHASE_WEIGHTS values
by playing real games (vs baseline bots AND self-play among top individuals)
and scoring win_rate - 2*self_death_rate. Best result is written to
weights.json, which main.py loads automatically at import time.

Usage:
    python3 tune_weights.py --minutes 480 --population 24
    python3 tune_weights.py --resume            # continue from checkpoint

Safe to Ctrl+C at any time — saves on exit.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_real_argv = sys.argv
sys.argv = [_real_argv[0]]  # run_games.py parses sys.argv at import time
import run_games as rg  # noqa: E402
sys.argv = _real_argv

import main as engine  # noqa: E402

import logging  # noqa: E402
logging.getLogger("battlesnake").setLevel(logging.CRITICAL)

CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_checkpoint.json")
BEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")

PHASES = [p.value for p in engine.GamePhase]
WEIGHT_KEYS = list(engine.PHASE_WEIGHTS[engine.GamePhase.EARLY].keys())

DIRS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}


# ---------------------------------------------------------------------------
# Individual = dict[phase_name][weight_name] -> float
# ---------------------------------------------------------------------------
def base_individual() -> dict:
    return {p.value: dict(w) for p, w in engine.PHASE_WEIGHTS.items()}


def mutate(ind: dict, rate: float = 0.35, strength: float = 0.35) -> dict:
    out = copy.deepcopy(ind)
    for phase in PHASES:
        for k in WEIGHT_KEYS:
            if random.random() > rate:
                continue
            v = out[phase][k]
            factor = 1.0 + random.uniform(-strength, strength)
            v = v * factor + random.gauss(0, 3.0)
            out[phase][k] = round(max(0.0, min(v, 500.0)), 2)
    return out


def crossover(a: dict, b: dict) -> dict:
    out = {}
    for phase in PHASES:
        out[phase] = {}
        for k in WEIGHT_KEYS:
            out[phase][k] = a[phase][k] if random.random() < 0.5 else b[phase][k]
    return out


# ---------------------------------------------------------------------------
# In-process game simulation (fast — reuses run_games.py internals directly)
# ---------------------------------------------------------------------------
def play_game(agents: list, seed: int, max_turns: int = 300) -> tuple[list[bool], list[int | None]]:
    """agents: list of objects with .get_name()/.start()/.move(). Returns
    (win_flags, death_turns) — death_turns[i] is None if agent i survived
    (won or still alive at max_turns)."""
    random.seed(seed)
    rg.HAZARD_CELLS = []
    rg.HAZARD_DMG = 0
    game_id = f"tune-{seed}-{random.random()}"
    
    result = rg._run_standalone_game(agents, game_id, verbose=False)
    
    win_flags = [False] * len(agents)
    if result["winner"] is not None:
        win_flags[result["winner"]] = True
        
    death_turns = [None] * len(agents)
    # Map deaths by name (works well for our uniquely named tuned agents)
    name_to_idx = {a.get_name(): i for i, a in enumerate(agents)}
    for d_event in result.get("deaths", []):
        name = d_event["agent"]
        if name in name_to_idx:
            idx = name_to_idx[name]
            if death_turns[idx] is None:
                death_turns[idx] = d_event["turn"]
                
    return win_flags, death_turns


class _TunedAgent:
    """Wraps main.TacticalEngine with a fixed weights dict, swapped into
    main.PHASE_WEIGHTS immediately before each move (sequential calls within
    one process -> safe)."""

    def __init__(self, weights: dict, tag: str):
        self.weights = {engine.GamePhase(p): w for p, w in weights.items()}
        self.tag = tag
        self.gid = None

    def get_name(self) -> str:
        return f"tuned-{self.tag}"

    def start(self, state) -> None:
        self.gid = f"{self.tag}-{id(state)}-{random.random()}"

    def move(self, state):
        data = state.model_dump() if hasattr(state, "model_dump") else state
        gid = f"{self.tag}-{data['game']['id']}"
        engine._update_enemy_memory(gid, data)
        food = engine._update_food_memory(gid, data)
        engine.PHASE_WEIGHTS.update(self.weights)
        d = engine.TacticalEngine.get_best_move(data, food, gid, time.monotonic() + engine.COMPUTE_BUDGET_S)

        class _R:
            move = d
        return _R()


EARLY_DEATH_TURN = 15  # dying before this turn is almost always a self-inflicted mistake


def eval_vs_baselines(weights: dict, n_games: int, seed_base: int) -> tuple[float, float]:
    wins, early_deaths = 0, 0
    for g in range(n_games):
        seed = seed_base + g
        if g % 2 == 0:
            opponents = [rg.RandomAgent(), rg.SafeFoodSeekingAgent(), rg.RandomAgent()]
        else:
            opponents = [rg.SafeFoodSeekingAgent(), rg.SafeFoodSeekingAgent(), rg.RandomAgent()]
        agents = [_TunedAgent(weights, "x")] + opponents
        win_flags, death_turns = play_game(agents, seed=seed)
        if win_flags[0]:
            wins += 1
        if death_turns[0] is not None and death_turns[0] < EARLY_DEATH_TURN:
            early_deaths += 1
    return wins / n_games, early_deaths / n_games


def eval_selfplay(pop_weights: list[dict], n_games_per_pair: int, seed_base: int) -> list[float]:
    """Round robin among a small elite set. Returns win_rate per individual."""
    n = len(pop_weights)
    wins = [0] * n
    played = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            for g in range(n_games_per_pair):
                seed = seed_base + i * 1000 + j * 10 + g
                agents = [_TunedAgent(pop_weights[i], f"a{i}"), _TunedAgent(pop_weights[j], f"b{j}")]
                win_flags, _ = play_game(agents, seed=seed)
                played[i] += 1
                played[j] += 1
                if win_flags[0]:
                    wins[i] += 1
                elif win_flags[1]:
                    wins[j] += 1
    return [wins[i] / played[i] if played[i] else 0.0 for i in range(n)]


def _worker_eval(args) -> tuple[int, float, float]:
    idx, weights, n_games, seed_base = args
    wr, early_death_rate = eval_vs_baselines(weights, n_games, seed_base)
    return idx, wr, early_death_rate


def _pool_init():
    logging.getLogger("battlesnake").setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Main evolutionary loop
# ---------------------------------------------------------------------------
def load_checkpoint():
    if os.path.isfile(CKPT_PATH):
        with open(CKPT_PATH) as f:
            data = json.load(f)
        return data["population"], data["gen"], data["best"], data["best_fitness"]
    return None


def save_checkpoint(population, gen, best, best_fitness):
    with open(CKPT_PATH, "w") as f:
        json.dump({"population": population, "gen": gen, "best": best, "best_fitness": best_fitness}, f)
    with open(BEST_PATH, "w") as f:
        json.dump(best, f, indent=2)


def main(args):
    ckpt = load_checkpoint() if args.resume else None
    if ckpt:
        population, gen, best, best_fitness = ckpt
        print(f"resumed gen={gen} best_fitness={best_fitness:.3f}")
    else:
        base = base_individual()
        population = [base] + [mutate(base, rate=0.5, strength=0.5) for _ in range(args.population - 1)]
        gen = 0
        best, best_fitness = base, -1.0

    deadline = time.monotonic() + args.minutes * 60
    pool = mp.Pool(processes=args.workers, initializer=_pool_init)

    def handle_sigint(signum, frame):
        print("\ninterrupted, saving checkpoint")
        save_checkpoint(population, gen, best, best_fitness)
        pool.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while time.monotonic() < deadline:
            gen += 1
            if best_fitness >= 0:
                population[0] = copy.deepcopy(best)
            
            seed_base = random.randint(0, 10**6)
            tasks = [(i, population[i], args.games_per_eval, seed_base) for i in range(len(population))]
            results = pool.map(_worker_eval, tasks)
            baseline_wr = [0.0] * len(population)
            early_death = [0.0] * len(population)
            for idx, wr, edr in results:
                baseline_wr[idx] = wr
                early_death[idx] = edr

            # win_rate - 2*early_death_rate: matches the spec's tuning
            # metric ("heavily penalize self-deaths").
            base_fitness = [baseline_wr[i] - 2.0 * early_death[i] for i in range(len(population))]

            elite_idx = sorted(range(len(population)), key=lambda i: -base_fitness[i])[: args.elite]
            elite_weights = [population[i] for i in elite_idx]
            selfplay_wr = eval_selfplay(elite_weights, args.selfplay_games, seed_base)

            fitness = list(base_fitness)
            for rank, idx in enumerate(elite_idx):
                fitness[idx] = 0.6 * base_fitness[idx] + 0.4 * selfplay_wr[rank]

            gen_best_idx = max(range(len(population)), key=lambda i: fitness[i])
            if fitness[gen_best_idx] > best_fitness:
                candidate = population[gen_best_idx]
                
                # Held-out validation
                val_wr, val_edr = eval_vs_baselines(candidate, args.games_per_eval, seed_base + 900000)
                val_fitness = val_wr - 2.0 * val_edr
                
                if val_fitness > best_fitness:
                    import subprocess
                    import shutil
                    temp_path = BEST_PATH + ".temp"
                    with open(temp_path, "w") as f:
                        json.dump(candidate, f, indent=2)
                    
                    if os.path.exists(BEST_PATH):
                        shutil.copy(BEST_PATH, BEST_PATH + ".bak")
                        
                    shutil.copy(temp_path, BEST_PATH)
                    rc = subprocess.call([sys.executable, "-m", "pytest", "test_seed8_regression.py", "-v"])
                    
                    if rc == 0:
                        best_fitness = fitness[gen_best_idx]
                        best = candidate
                        save_checkpoint(population, gen, best, best_fitness)
                        print(f"gen {gen}: NEW BEST fitness={best_fitness:.3f} (val_fitness={val_fitness:.3f})")
                    else:
                        print(f"gen {gen}: REJECTED by regression gate")
                        if os.path.exists(BEST_PATH + ".bak"):
                            shutil.copy(BEST_PATH + ".bak", BEST_PATH)
                        # Reset fitness so it doesn't try again right away
                        fitness[gen_best_idx] = -999.0
                else:
                    print(f"gen {gen}: REJECTED by validation (val_fitness={val_fitness:.3f} <= best={best_fitness:.3f})")
                    fitness[gen_best_idx] = -999.0
            else:
                print(f"gen {gen}: best_this_gen={fitness[gen_best_idx]:.3f} (all-time={best_fitness:.3f})")

            ranked = sorted(range(len(population)), key=lambda i: -fitness[i])
            survivors = [population[i] for i in ranked[: max(4, len(population) // 3)]]
            new_pop = list(survivors)
            while len(new_pop) < args.population:
                a, b = random.sample(survivors, 2) if len(survivors) >= 2 else (survivors[0], survivors[0])
                child = mutate(crossover(a, b))
                new_pop.append(child)
            population = new_pop

            save_checkpoint(population, gen, best, best_fitness)
    finally:
        pool.close()
        pool.join()

    print(f"done. gens={gen} best_fitness={best_fitness:.3f}  -> {BEST_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=480.0)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--elite", type=int, default=5)
    ap.add_argument("--games-per-eval", type=int, default=40)
    ap.add_argument("--selfplay-games", type=int, default=6)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--resume", action="store_true")
    main(ap.parse_args())
