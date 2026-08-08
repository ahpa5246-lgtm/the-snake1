"""
test_seed8_regression.py — permanent regression benchmark.

Origin: local testing with `run_games.py --seed 8 --no-hisss` reproduced a
death for our snake very early in the game (well before turn 30), caused by
Layer 1's survival filter being either too strict (vetoing every direction
and falling through to a blind fallback) or too lenient (missing a real
trap). This test locks in the fix.

Rule going forward: any change to main.py must keep this test green before
being accepted (per the task spec, section 0).

Run directly:  python3 test_seed8_regression.py
Run via pytest: pytest test_seed8_regression.py -v
"""

from __future__ import annotations

import os
import sys
import random

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_SURVIVAL_TURN = 30

# Our snake's display name, as returned by agent_adapter.ThuebanAgent.get_name().
# Imported (rather than hardcoded) so the test still works if the identity
# ever changes.
sys.path.insert(0, REPO_DIR)
from agent_adapter import ThuebanAgent  # noqa: E402

OUR_NAME = ThuebanAgent().get_name()


def _run_seed(seed: int, games: int = 1, extra_args: list[str] | None = None) -> list[dict]:
    """Runs standalone game directly and returns structured JSON results."""
    import run_games
    from run_games import SafeFoodSeekingAgent, RandomAgent
    from agent_adapter import ThuebanAgent, move_latency_ms

    random.seed(seed)
    run_games.HAZARD_CELLS = []
    run_games.HAZARD_DMG = 14
    move_latency_ms.clear()

    all_results = []
    for g in range(games):
        agents = [ThuebanAgent(), SafeFoodSeekingAgent(), RandomAgent(), RandomAgent()]
        game_id = f"test-{seed}-{g}"
        res = run_games._run_standalone_game(agents, game_id, verbose=False)
        all_results.append(res)
    return all_results

def _our_death_events(results: list[dict]) -> list[dict]:
    events = []
    for r in results:
        for d in r.get("deaths", []):
            if d["agent"] == OUR_NAME:
                events.append(d)
    return events


def test_seed8_survives_past_turn_30():
    """
    THE regression test (task spec, section 0): with `--seed 8`, our snake
    must not die before turn 30. It's fine if the game ends earlier because
    we *won* (all death turns list empty) — only an early death fails this.
    """
    results = _run_seed(seed=8)
    death_events = _our_death_events(results)
    early_deaths = [d for d in death_events if d["turn"] < MIN_SURVIVAL_TURN]

    assert not early_deaths, (
        f"Our snake died before turn {MIN_SURVIVAL_TURN} on seed=8 "
        f"(death events found: {early_deaths})."
    )


def test_seed8_no_exceptions():
    """A broader sanity check alongside the main benchmark: seed=8 must run
    to completion without the engine ever hitting its top-level exception
    handler (which silently falls back to 'up')."""
    results = _run_seed(seed=8)
    for r in results:
        assert "error" not in r, f"get_best_move raised an exception:\n{r.get('error')}"


def test_no_self_inflicted_death_before_turn_30_multiseed():
    """
    Bonus sweep (not required by the spec, but cheap insurance): across a
    fixed spread of seeds, our snake should never die from an avoidable
    self-inflicted cause (its own body) before turn 30. H2H / starvation /
    hazard deaths are excluded — those can be legitimate fog-of-war outcomes
    (colliding with a snake we genuinely could not see), not logic bugs.
    """
    bad = []
    for seed in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        results = _run_seed(seed=seed, games=2)
        death_events = _our_death_events(results)
        
        for d in death_events:
            if d["cause"] == "body-collision" and d["turn"] < MIN_SURVIVAL_TURN:
                bad.append((seed, d["turn"]))

    assert not bad, f"Self-inflicted early deaths found (seed, turn): {bad}"

def test_latency_budget():
    from agent_adapter import move_latency_ms
    from main import TacticalEngine
    results = _run_seed(seed=42, games=2)
    budget = TacticalEngine.COMPUTE_BUDGET_S * 1000
    over_budget = [ms for ms in move_latency_ms if ms > budget]
    assert not over_budget, f"Moves exceeded {budget}ms budget: {over_budget}"


if __name__ == "__main__":
    tests = [
        test_seed8_survives_past_turn_30,
        test_seed8_no_exceptions,
        test_no_self_inflicted_death_before_turn_30_multiseed,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}\n{e}\n")
    if failures:
        print(f"\n{failures}/{len(tests)} test(s) failed.")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
