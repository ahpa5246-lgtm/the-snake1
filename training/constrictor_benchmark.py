"""Mirrored 4-player Constrictor benchmark against the pre-change main.py.

Runs with Hisss (not the approximate standalone Blackout simulator).
Does not train, publish, or promote a policy. Report includes every game,
move invalidity, decision overruns and score; no statistical proof is implied.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def baseline_module(ref: str):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:main.py"], cwd=ROOT, text=True,
    )
    module = types.ModuleType("frozen_baseline_main")
    module.__file__ = str(ROOT / "main.py")
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


class NoNeural:
    @staticmethod
    def scores(_raw, _legal, _deadline):
        return {}


def make_config(hisss):
    config = hisss.standard_config()
    config.constrictor = True
    config.food_spawn_chance = 0
    config.min_food = 0
    config.all_actions_legal = False
    config.init_snake_pos = {
        0: [[1, 1], [1, 2], [1, 3]],
        1: [[9, 9], [9, 8], [9, 7]],
        2: [[1, 9], [1, 8], [1, 7]],
        3: [[9, 1], [9, 2], [9, 3]],
    }
    config.init_food_pos = []
    return config


def run_match(hisss, candidate, baseline, seed: int, mirror: int, max_turns: int):
    env = hisss.BattleSnakeGame(make_config(hisss))
    candidate_slot, baseline_slot = ((0, 1) if mirror == 0 else (1, 0))
    mapping = {"up": hisss.UP, "down": hisss.DOWN,
               "left": hisss.LEFT, "right": hisss.RIGHT}
    rng = random.Random(seed)
    engines = {candidate_slot: candidate, baseline_slot: baseline}
    invalid = {"candidate": 0, "baseline": 0}
    times = {"candidate": [], "baseline": []}
    game_tag = f"benchmark-{seed}-{mirror}"
    elimination = {}
    was_alive = set(env.players_alive())
    turn = 0
    while not env.is_terminal() and turn < max_turns:
        actions = []
        for slot in env.players_at_turn():
            legal = tuple(env.available_actions(slot))
            if slot in engines:
                engine = engines[slot]
                label = "candidate" if slot == candidate_slot else "baseline"
                raw = json.loads(hisss.to_battlesnake_json(env, slot))
                # Hisss config flags, not JSON presentation names, define the
                # physics; set our policy's explicit ruleset discriminator.
                raw.setdefault("game", {}).setdefault("ruleset", {})["name"] = "constrictor"
                started = time.monotonic()
                chosen = engine.TacticalEngine.get_best_move(
                    raw, set(), f"{game_tag}-{label}",
                    started + engine.COMPUTE_BUDGET_S, advisor=NoNeural,
                )
                elapsed = (time.monotonic() - started) * 1000.0
                times[label].append(elapsed)
                action = mapping.get(chosen)
                if action not in legal:
                    invalid[label] += 1
                    action = legal[0] if legal else hisss.UP
            else:
                action = rng.choice(legal) if legal else hisss.UP
            actions.append(action)
        env.step(actions=tuple(actions))
        turn += 1
        alive = set(env.players_alive())
        for slot in was_alive - alive:
            elimination[slot] = turn
        was_alive = alive

    alive = set(env.players_alive())
    candidate_turn = turn if candidate_slot in alive else elimination.get(candidate_slot, turn)
    baseline_turn = turn if baseline_slot in alive else elimination.get(baseline_slot, turn)
    # A tied knockout is not counted as an improvement.
    result = "candidate" if candidate_turn > baseline_turn else (
        "baseline" if baseline_turn > candidate_turn else "draw"
    )
    for engine, label in ((candidate, "candidate"), (baseline, "baseline")):
        engine._game_memory.pop(f"{game_tag}-{label}", None)
    return {
        "seed": seed, "mirror": mirror, "outcome": result,
        "turns": turn, "candidate_survival": candidate_turn,
        "baseline_survival": baseline_turn, "invalid": invalid,
        "decision_ms": {k: times[k] for k in times},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ref", default="origin/main")
    parser.add_argument("--games", type=int, default=16,
                        help="Must be a positive even number (mirrored pairs).")
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--output", default="constrictor-benchmark.json")
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error("--games must be a positive even number")
    import hisss
    import main as candidate
    baseline = baseline_module(args.baseline_ref)
    matches = [
        run_match(hisss, candidate, baseline, args.seed + i, mirror,
                  args.max_turns)
        for i in range(args.games // 2)
        for mirror in (0, 1)
    ]
    decision_times = {label: sorted(
        sample for match in matches for sample in match["decision_ms"][label]
    ) for label in ("candidate", "baseline")}
    latency = {
        label: {
            "p95_ms": round(samples[min(len(samples)-1, int(len(samples)*0.95))], 2)
            if samples else None,
            "budget_overruns": sum(ms > 200 for ms in samples),
            "moves": len(samples),
        } for label, samples in decision_times.items()
    }
    counts = {label: sum(match["outcome"] == label for match in matches)
              for label in ("candidate", "baseline", "draw")}
    report = {
        "protocol": "hisss-constrictor-mirrored-v1",
        "baseline_ref": args.baseline_ref,
        "seed": args.seed, "games": args.games, "max_turns": args.max_turns,
        "counts": counts,
        "candidate_win_share": round(
            (counts["candidate"] + counts["draw"] / 2) / args.games, 4),
        "invalid_moves": {
            label: sum(match["invalid"][label] for match in matches)
            for label in ("candidate", "baseline")},
        "latency": latency,
        "matches": [{key: value for key, value in match.items()
                     if key != "decision_ms"} for match in matches],
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "matches"}, indent=2))


if __name__ == "__main__":
    main()
