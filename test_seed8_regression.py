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
import re
import subprocess
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_SURVIVAL_TURN = 30

# Our snake's display name, as returned by agent_adapter.ThuebanAgent.get_name().
# Imported (rather than hardcoded) so the test still works if the identity
# ever changes.
sys.path.insert(0, REPO_DIR)
from agent_adapter import ThuebanAgent  # noqa: E402

OUR_NAME = ThuebanAgent().get_name()


def _run_seed(seed: int, games: int = 1, extra_args: list[str] | None = None) -> str:
    """Runs run_games.py with a fixed seed and returns combined stdout+stderr."""
    cmd = [
        sys.executable, "run_games.py",
        "--games", str(games),
        "--seed", str(seed),
        "--no-hisss",
        "--verbose",
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=REPO_DIR, timeout=120,
    )
    return result.stdout + result.stderr, result.returncode


def _our_death_turns(log_text: str) -> list[int]:
    """Every turn our snake died, across however many games were in the log."""
    pattern = re.compile(rf"\[{re.escape(OUR_NAME)}\]\s+died turn\s+(\d+):")
    return [int(m.group(1)) for m in pattern.finditer(log_text)]


def test_seed8_survives_past_turn_30():
    """
    THE regression test (task spec, section 0): with `--seed 8`, our snake
    must not die before turn 30. It's fine if the game ends earlier because
    we *won* (all death turns list empty) — only an early death fails this.
    """
    log_text, rc = _run_seed(seed=8)
    assert rc == 0, f"run_games.py exited with code {rc}:\n{log_text}"

    death_turns = _our_death_turns(log_text)
    early_deaths = [t for t in death_turns if t < MIN_SURVIVAL_TURN]

    assert not early_deaths, (
        f"Our snake died before turn {MIN_SURVIVAL_TURN} on seed=8 "
        f"(death turns found: {death_turns}).\n\nFull log:\n{log_text}"
    )


def test_seed8_no_exceptions():
    """A broader sanity check alongside the main benchmark: seed=8 must run
    to completion without the engine ever hitting its top-level exception
    handler (which silently falls back to 'up')."""
    log_text, rc = _run_seed(seed=8)
    assert rc == 0, f"run_games.py exited with code {rc}:\n{log_text}"
    assert "Exception in get_best_move" not in log_text, (
        f"get_best_move raised at least once on seed=8:\n{log_text}"
    )


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
        log_text, rc = _run_seed(seed=seed, games=2)
        assert rc == 0, f"seed={seed}: run_games.py exited with code {rc}:\n{log_text}"

        pattern = re.compile(
            rf"\[{re.escape(OUR_NAME)}\]\s+died turn\s+(\d+):\s+body collision"
        )
        for m in pattern.finditer(log_text):
            turn = int(m.group(1))
            if turn < MIN_SURVIVAL_TURN:
                bad.append((seed, turn))

    assert not bad, f"Self-inflicted early deaths found (seed, turn): {bad}"


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
