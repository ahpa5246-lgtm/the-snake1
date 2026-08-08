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
            out[phase][k] = round(max(0.0, v), 2)
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
def _new_snakes(n: int, agents: list) -> list:
    positions = [(1, 1), (9, 9), (1, 9), (9, 1)][:n]
    return [rg.Snake(id=f"s{i}", name=agents[i].get_name(), body=[p, p, p]) for i, p in enumerate(positions)]


def play_game(agents: list, seed: int, max_turns: int = 300) -> tuple[list[bool], list[int | None]]:
    """agents: list of objects with .get_name()/.start()/.move(). Returns
    (win_flags, death_turns) — death_turns[i] is None if agent i survived
    (won or still alive at max_turns)."""
    random.seed(seed)
    game_id = f"tune-{seed}-{random.random()}"
    snakes = _new_snakes(len(agents), agents)
    death_turns: list[int | None] = [None] * len(agents)
    food: list = []
    occ0 = {s.body[0] for s in snakes}
    for _ in range(rg.FOOD_MIN):
        p = rg._random_empty(occ0 | set(food))
        if p:
            food.append(p)

    for i, a in enumerate(agents):
        try:
            a.start(rg._wrap(rg._mk_state(game_id, 0, snakes, food, i)))
        except Exception:
            pass

    turn = 0
    while True:
        turn += 1
        alive = [i for i, s in enumerate(snakes) if s.alive]
        if len(alive) <= 1 or turn > max_turns:
            break
        if len(food) < rg.FOOD_MIN or random.random() < rg.FOOD_SPAWN_PROB:
            all_occ = {seg for s in snakes if s.alive for seg in s.body} | set(food)
            p = rg._random_empty(all_occ)
            if p:
                food.append(p)

        moves = {}
        for i in alive:
            try:
                mv = agents[i].move(rg._wrap(rg._mk_state(game_id, turn, snakes, food, i)))
                d = str(mv.move).lower() if hasattr(mv, "move") else str(mv).lower()
                d = next((k for k in DIRS if k in d), "up")
            except Exception:
                d = "up"
            moves[i] = d

        new_heads = {i: (snakes[i].head[0] + DIRS[moves[i]][0], snakes[i].head[1] + DIRS[moves[i]][1]) for i in alive}

        for i in alive:
            nx, ny = new_heads[i]
            if not (0 <= nx < rg.WIDTH and 0 <= ny < rg.HEIGHT):
                snakes[i].alive = False; death_turns[i] = death_turns[i] or turn

        alive = [i for i, s in enumerate(snakes) if s.alive]
        body_cells = rg._build_occupied_sim([snakes[i] for i in alive])
        for i in alive:
            if new_heads[i] in body_cells:
                snakes[i].alive = False; death_turns[i] = death_turns[i] or turn

        alive = [i for i, s in enumerate(snakes) if s.alive]
        head_pos: dict = {}
        for i in alive:
            head_pos.setdefault(new_heads[i], []).append(i)
        for pos, claimants in head_pos.items():
            if len(claimants) < 2:
                continue
            max_len = max(snakes[i].length for i in claimants)
            for i in claimants:
                if snakes[i].length < max_len:
                    snakes[i].alive = False; death_turns[i] = death_turns[i] or turn
                else:
                    peers = [j for j in claimants if j != i and snakes[j].length == max_len]
                    if peers:
                        snakes[i].alive = False; death_turns[i] = death_turns[i] or turn

        alive = [i for i, s in enumerate(snakes) if s.alive]
        for i in alive:
            ate = new_heads[i] in food
            snakes[i].body.insert(0, new_heads[i])
            if not ate:
                snakes[i].body.pop()
            else:
                snakes[i].health = rg.START_HEALTH
                food.remove(new_heads[i])

        hazard_set = set(rg.HAZARD_CELLS)
        for i in alive:
            snakes[i].health -= rg.HUNGER_PER_TURN
            if snakes[i].head in hazard_set:
                snakes[i].health -= rg.HAZARD_DMG
            if snakes[i].health <= 0:
                snakes[i].alive = False; death_turns[i] = death_turns[i] or turn

    alive = [i for i, s in enumerate(snakes) if s.alive]
    winner = alive[0] if len(alive) == 1 else None
    return [i == winner for i in range(len(agents))], death_turns


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
        opponents = [rg.RandomAgent(), rg.SafeFoodSeekingAgent(), rg.RandomAgent()]
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
                best_fitness = fitness[gen_best_idx]
                best = population[gen_best_idx]
                save_checkpoint(population, gen, best, best_fitness)
                print(f"gen {gen}: NEW BEST fitness={best_fitness:.3f} (baseline_wr={baseline_wr[gen_best_idx]:.2f})")
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
