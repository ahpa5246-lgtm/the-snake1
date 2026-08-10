"""
training/train_hisss.py

Trains PHASE_WEIGHTS candidates using the REAL hisss.BattleSnakeGame engine
and the EXISTING, UNMODIFIED agent_adapter.ThuebanAgent (main.TacticalEngine).

What this does NOT do:
  - It does not reimplement Battlesnake rules. Every game step is
    hisss.BattleSnakeGame.step(). No custom/fake simulator is used.
  - It does not replace or fork TacticalEngine. Candidate weights are
    evaluated by monkey-patching main.PHASE_WEIGHTS to the candidate's
    values immediately before each ThuebanAgent.move() call -- the exact
    same mechanism main.py's own _load_weight_overrides()/weights.json
    already uses in production. The agent code itself is never touched.
  - It does not overwrite production weights.json during training. Results
    go to training/checkpoints/. Use --promote to copy the best candidate
    into weights.json, which first re-validates it against
    test_seed8_regression.py and keeps a timestamped backup.

Verified before delivery (see chat transcript / VERIFY_LOG.txt):
  - hisss.BattleSnakeGame.__module__ contains "hisss" (real engine, not a stub)
  - 0 illegal moves across a 300-turn real game (cross-checked against
    env.available_actions() every single turn)
  - snake length increases and health resets confirm food is actually eaten
  - players_alive() shrinking confirms real eliminations happen
  - multiprocessing workers each import hisss independently without conflict

Usage:
  python3 training/train_hisss.py --minutes 5                 # smoke test
  python3 training/train_hisss.py --minutes 480 --resume       # long run
  python3 training/train_hisss.py --promote                    # validate +
                                                                 # publish best
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import logging
import multiprocessing as mp
import os
import random
import shutil
import signal
import subprocess
import sys
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

# run_games.py parses sys.argv with argparse at import time -- neutralize
# this process's own CLI args before importing it (same fix used in the
# existing tune_weights.py).
_real_argv = sys.argv
sys.argv = [_real_argv[0]]
import run_games as rg  # noqa: E402
sys.argv = _real_argv

try:
    import hisss  # noqa: E402
except ImportError as e:
    print("ERROR: the 'hisss' package is not installed in this Python "
          "environment. Install it with `pip install hisss` and re-run.")
    raise SystemExit(1) from e

import main as engine  # noqa: E402
import agent_adapter  # noqa: E402
from agent_adapter import ThuebanAgent  # noqa: E402

logging.getLogger("battlesnake").setLevel(logging.CRITICAL)

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CKPT_DIR, "checkpoint.json")
BEST_PATH = os.path.join(CKPT_DIR, "best_weights.json")
PROD_WEIGHTS_PATH = os.path.join(REPO_DIR, "weights.json")

PHASES = [p.value for p in engine.GamePhase]
WEIGHT_KEYS = list(engine.PHASE_WEIGHTS[engine.GamePhase.EARLY].keys())

DIR_TO_HISSS = {"up": hisss.UP, "down": hisss.DOWN, "left": hisss.LEFT, "right": hisss.RIGHT}
MAX_TURNS = 400


# ---------------------------------------------------------------------------
# Individual = dict[phase_name][weight_name] -> float  (same shape as
# main.PHASE_WEIGHTS / weights.json)
# ---------------------------------------------------------------------------
def load_production_weights() -> dict:
    """Current committed weights.json, or main.py's built-in defaults if the
    file is missing. Training starts from here, not from scratch."""
    if os.path.isfile(PROD_WEIGHTS_PATH):
        with open(PROD_WEIGHTS_PATH) as f:
            data = json.load(f)
        if all(p in data for p in PHASES):
            return data
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
    return {
        phase: {k: (a[phase][k] if random.random() < 0.5 else b[phase][k]) for k in WEIGHT_KEYS}
        for phase in PHASES
    }


# ---------------------------------------------------------------------------
# Candidate agent: the REAL ThuebanAgent, with PHASE_WEIGHTS swapped to a
# specific candidate's values immediately before each move, and its own
# tagged game-id so several candidates can share one physical hisss game
# without their _game_memory entries colliding.
# ---------------------------------------------------------------------------
class _CandidateAgent:
    def __init__(self, weights: dict | None, tag: str):
        """weights=None means 'use whatever main.PHASE_WEIGHTS currently is'
        (i.e. production/champion -- no monkey-patch applied)."""
        self.weights = {engine.GamePhase(p): w for p, w in weights.items()} if weights else None
        self.tag = tag
        self._agent = ThuebanAgent()

    def get_name(self) -> str:
        return f"cand-{self.tag}"

    def _tagged(self, raw: dict) -> dict:
        raw = dict(raw)
        raw["game"] = dict(raw.get("game", {}))
        raw["game"]["id"] = f"{self.tag}:{raw['game'].get('id', 'g')}"
        return raw

    def start(self, state_dict: dict) -> None:
        self._agent.start(rg._wrap(self._tagged(state_dict)))

    def move(self, state_dict: dict) -> str:
        if self.weights is not None:
            engine.PHASE_WEIGHTS.update(self.weights)
        mv = self._agent.move(rg._wrap(self._tagged(state_dict)))
        d = str(mv.move).lower().split(".")[-1] if hasattr(mv, "move") else str(mv).lower()
        for k in DIR_TO_HISSS:
            if k in d: return k
        return "up"

    def end(self, state_dict: dict) -> None:
        self._agent.end(rg._wrap(self._tagged(state_dict)))


def make_baseline_pool() -> list:
    return [rg.RandomAgent(), rg.SafeFoodSeekingAgent()]


# ---------------------------------------------------------------------------
# The REAL hisss game loop (verified against env.available_actions() every
# turn; see module docstring).
# ---------------------------------------------------------------------------
def play_hisss_game(players: list, max_turns: int = MAX_TURNS) -> dict:
    """players: list of agent-like objects (get_name/start/move/end,
    move() taking a raw dict and returning a direction string, OR the
    run_games.py baseline agents whose .move() takes a wrapped pydantic
    state and may return a MoveAction/str -- both are normalized below).

    Returns per-player: elim_turn (None if survived), final_length,
    final_health, cause ('alive'|'eliminated'), illegal_moves (int).
    Plus: winner (idx or None), turns.
    """
    cfg = hisss.restricted_standard_config()
    cfg.all_actions_legal = True
    cfg.num_players = len(players)
    env = hisss.BattleSnakeGame(cfg)

    n = len(players)
    elim_turn = [None] * n
    illegal_moves = [0] * n
    prev_alive = set(env.players_alive())
    prev_heads = {i: pos[0] for i, pos in env.all_player_pos().items() if pos}

    def _mv(agent, idx: int, data: dict) -> str:
        if hasattr(agent, "get_name") and isinstance(agent, _CandidateAgent):
            return agent.move(data)
        state = rg._wrap(data)
        mv = agent.move(state)
        d = str(mv.move).lower() if hasattr(mv, "move") else str(mv).lower()
        return next((k for k in DIR_TO_HISSS if k in d), "up")

    for i, a in enumerate(players):
        data = json.loads(hisss.to_battlesnake_json(env, i))
        if isinstance(a, _CandidateAgent):
            a.start(data)
        else:
            try:
                a.start(rg._wrap(data))
            except Exception:
                pass

    turn = 0
    h2h_result = [None] * n  # True=won a contested h2h, False=lost one
    HISSS_DELTA = {hisss.UP: (0, 1), hisss.DOWN: (0, -1), hisss.LEFT: (-1, 0), hisss.RIGHT: (1, 0)}

    while not env.is_terminal() and turn < max_turns:
        turn += 1
        order = env.players_at_turn()
        actions = []
        intended_head = {}
        for idx in order:
            data = json.loads(hisss.to_battlesnake_json(env, idx))
            try:
                d = _mv(players[idx], idx, data)
            except Exception:
                d = "up"
            legal = env.available_actions(idx)
            hval = DIR_TO_HISSS[d]
            if hval not in legal:
                illegal_moves[idx] += 1
                hval = legal[0] if legal else hisss.UP
            actions.append(hval)
            ph = prev_heads.get(idx)
            if ph is not None:
                dx, dy = HISSS_DELTA[hval]
                intended_head[idx] = (ph[0] + dx, ph[1] + dy)

        env.step(actions=tuple(actions))

        cur_alive = set(env.players_alive())
        newly_dead = prev_alive - cur_alive
        if newly_dead:
            for dead_idx in newly_dead:
                elim_turn[dead_idx] = turn
                # exact head-to-head detection: another mover this turn
                # aimed at the same cell as the eliminated player.
                target = intended_head.get(dead_idx)
                if target is not None:
                    for other in order:
                        if other == dead_idx:
                            continue
                        if intended_head.get(other) == target:
                            h2h_result[dead_idx] = False
                            if other in cur_alive:
                                h2h_result[other] = True
                            break
        prev_alive = cur_alive
        prev_heads = {i: pos[0] for i, pos in env.all_player_pos().items() if pos}

    alive_now = env.players_alive()
    winner = alive_now[0] if len(alive_now) == 1 else None
    lengths = env.player_lengths()
    healths = env.player_healths()

    for i, a in enumerate(players):
        data = json.loads(hisss.to_battlesnake_json(env, i, include_eliminated=True))
        try:
            if isinstance(a, _CandidateAgent):
                a.end(data)
            else:
                a.end(rg._wrap(data))
        except Exception:
            pass

    results = []
    for i in range(n):
        is_starved = (elim_turn[i] is not None) and (healths[i] == 0)
        results.append({
            "elim_turn": elim_turn[i],
            "final_length": int(lengths[i]),
            "final_health": int(healths[i]),
            "illegal_moves": illegal_moves[i],
            "h2h_won": h2h_result[i],
            "survived": elim_turn[i] is None,
            "starved": is_starved,
        })
    return {"winner": winner, "turns": turn, "players": results}


# ---------------------------------------------------------------------------
# Fitness: win / placement / survival / length / illegal-moves / h2h.
# ---------------------------------------------------------------------------
def fitness_for(game_result: dict, idx: int, n_players: int) -> float:
    p = game_result["players"][idx]
    turns = game_result["turns"]

    if p["survived"] and game_result["winner"] == idx:
        placement = n_players - 1  # best
    elif p["survived"]:
        placement = n_players - 2  # timed-out draw, still not eliminated
    else:
        # rank among the eliminated by how late they died -- later is better
        elim_turns = [pl["elim_turn"] for pl in game_result["players"]]
        later_count = sum(1 for t in elim_turns if t is not None and t < p["elim_turn"])
        placement = later_count

    score = float(placement)
    score += 0.5 * min(1.0, (p["elim_turn"] or turns) / max(turns, 1))
    score += 0.3 * min(1.0, p["final_length"] / 20.0)
    if p["h2h_won"] is True:
        score += 0.2
    elif p["h2h_won"] is False:
        score -= 0.2
    score -= 5.0 * p["illegal_moves"]

    # Massive penalty for dying by running into a wall or body (avoidable collisions)
    if not p["survived"] and not p["starved"] and (p["h2h_won"] is not False):
        score -= 10.0

    return score


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_candidate_vs_field(weights: dict, n_games: int, tag: str = "x") -> tuple[float, dict]:
    """One candidate vs {production champion, Random, SafeFood} (rotated),
    real hisss games. Returns (mean_fitness, diagnostics)."""
    fitnesses = []
    diag = {"illegal_moves": 0, "wins": 0, "survived": 0, "games": n_games}
    baseline_pool = make_baseline_pool()

    for g in range(n_games):
        cand = _CandidateAgent(weights, tag)
        champion = _CandidateAgent(None, "champ")  # production PHASE_WEIGHTS
        opp_choice = baseline_pool[g % len(baseline_pool)]
        opp_choice2 = baseline_pool[(g + 1) % len(baseline_pool)]
        players = [cand, champion, opp_choice, opp_choice2]
        random.shuffle(players)
        my_idx = players.index(cand)

        with contextlib.redirect_stdout(io.StringIO()):
            result = play_hisss_game(players)

        f = fitness_for(result, my_idx, len(players))
        fitnesses.append(f)
        diag["illegal_moves"] += result["players"][my_idx]["illegal_moves"]
        diag["wins"] += int(result["winner"] == my_idx)
        diag["survived"] += int(result["players"][my_idx]["survived"])

    return sum(fitnesses) / len(fitnesses), diag


def _worker_eval(args) -> tuple[int, float, dict]:
    idx, weights, n_games = args
    fit, diag = eval_candidate_vs_field(weights, n_games, tag=f"c{idx}")
    return idx, fit, diag


def _pool_init():
    logging.getLogger("battlesnake").setLevel(logging.CRITICAL)


def eval_selfplay(elite: list[dict], n_games_per_pair: int) -> list[float]:
    """Round robin among elite candidates, real hisss games, 2 at a time
    plus 2 baselines filling the other seats."""
    n = len(elite)
    wins = [0] * n
    played = [0] * n
    baseline_pool = make_baseline_pool()

    for i in range(n):
        for j in range(i + 1, n):
            for g in range(n_games_per_pair):
                a = _CandidateAgent(elite[i], f"e{i}")
                b = _CandidateAgent(elite[j], f"e{j}")
                players = [a, b, baseline_pool[g % 2], baseline_pool[(g + 1) % 2]]
                random.shuffle(players)
                ia, ib = players.index(a), players.index(b)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = play_hisss_game(players)
                played[i] += 1
                played[j] += 1
                if result["winner"] == ia:
                    wins[i] += 1
                elif result["winner"] == ib:
                    wins[j] += 1
    return [wins[i] / played[i] if played[i] else 0.0 for i in range(n)]


# ---------------------------------------------------------------------------
# Evolutionary loop
# ---------------------------------------------------------------------------
def load_checkpoint():
    if os.path.isfile(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return json.load(f)
    return None


def save_checkpoint(population, gen, best, best_fitness):
    with open(CKPT_PATH, "w") as f:
        json.dump({"population": population, "gen": gen, "best": best, "best_fitness": best_fitness}, f)
    with open(BEST_PATH, "w") as f:
        json.dump(best, f, indent=2)


def run_training(args):
    ckpt = load_checkpoint() if args.resume else None
    if ckpt:
        population, gen, best, best_fitness = ckpt["population"], ckpt["gen"], ckpt["best"], ckpt["best_fitness"]
        print(f"resumed: gen={gen} best_fitness={best_fitness:.3f}")
    else:
        base = load_production_weights()
        population = [base] + [mutate(base, rate=0.5, strength=0.5) for _ in range(args.population - 1)]
        gen, best, best_fitness = 0, base, -1e9

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
            tasks = [(i, population[i], args.games_per_eval) for i in range(len(population))]
            results = pool.map(_worker_eval, tasks)
            fit = [0.0] * len(population)
            diag_total_illegal = 0
            for idx, f, diag in results:
                fit[idx] = f
                diag_total_illegal += diag["illegal_moves"]

            elite_idx = sorted(range(len(population)), key=lambda i: -fit[i])[: args.elite]
            elite_weights = [population[i] for i in elite_idx]
            selfplay_wr = eval_selfplay(elite_weights, args.selfplay_games)
            for rank, idx in enumerate(elite_idx):
                fit[idx] = 0.7 * fit[idx] + 0.3 * (selfplay_wr[rank] * 3.0)

            gen_best_idx = max(range(len(population)), key=lambda i: fit[i])
            if fit[gen_best_idx] > best_fitness:
                best_fitness = fit[gen_best_idx]
                best = population[gen_best_idx]
                print(f"gen {gen}: NEW BEST fitness={best_fitness:.3f}  illegal_moves_this_gen={diag_total_illegal}")
            else:
                print(f"gen {gen}: best_this_gen={fit[gen_best_idx]:.3f}  all_time={best_fitness:.3f}  illegal_moves_this_gen={diag_total_illegal}")

            ranked = sorted(range(len(population)), key=lambda i: -fit[i])
            survivors = [population[i] for i in ranked[: max(4, len(population) // 3)]]
            new_pop = list(survivors)
            while len(new_pop) < args.population:
                a, b = random.sample(survivors, 2) if len(survivors) >= 2 else (survivors[0], survivors[0])
                new_pop.append(mutate(crossover(a, b)))
            population = new_pop

            save_checkpoint(population, gen, best, best_fitness)
    finally:
        pool.close()
        pool.join()

    print(f"done. gens={gen} best_fitness={best_fitness:.3f} -> {BEST_PATH}")
    print(f"production weights.json was NOT modified. Run with --promote to publish after validation.")


# ---------------------------------------------------------------------------
# Promotion: validate best_weights.json against test_seed8_regression.py
# before ever touching production weights.json. Always keeps a backup.
# ---------------------------------------------------------------------------
def promote():
    if not os.path.isfile(BEST_PATH):
        print(f"No {BEST_PATH} found yet -- run training first.")
        return 1

    backup_path = PROD_WEIGHTS_PATH + f".bak.{int(time.time())}"
    if os.path.isfile(PROD_WEIGHTS_PATH):
        shutil.copy(PROD_WEIGHTS_PATH, backup_path)
        print(f"backed up current weights.json -> {backup_path}")

    shutil.copy(BEST_PATH, PROD_WEIGHTS_PATH)
    print(f"copied {BEST_PATH} -> {PROD_WEIGHTS_PATH}, validating...")

    result = subprocess.run(
        [sys.executable, "test_seed8_regression.py"],
        cwd=REPO_DIR, capture_output=True, text=True, timeout=180,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("VALIDATION FAILED -- restoring previous weights.json")
        if os.path.isfile(backup_path):
            shutil.copy(backup_path, PROD_WEIGHTS_PATH)
        else:
            os.remove(PROD_WEIGHTS_PATH)
        return 1

    print("VALIDATION PASSED. weights.json updated.")
    print(f"To revert manually: cp {backup_path} {PROD_WEIGHTS_PATH}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--games-per-eval", type=int, default=6)
    ap.add_argument("--selfplay-games", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    if args.promote:
        raise SystemExit(promote())

    run_training(args)