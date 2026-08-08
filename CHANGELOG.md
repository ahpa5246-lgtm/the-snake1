# Battlesnake Blackout 2026 - Evolution Changelog

## V9.0 - APEX PREDATOR (Active)

### Engine Tuning & Weights
- **Elitism & Self-Play**: Introduced held-out validation and self-play weighting for robust weight evolution in `tune_weights.py`.
- **Weights Updated**: Continuous improvements gated by `test_seed8_regression.py`.

### Logic Fixes & Survival Improvements
- **Memory Fixes**: Fixed `_new_mem_entry` key errors and payload validation bugs on `on_move`.
- **Regression Net**: Established `test_seed8_regression.py` as a non-negotiable 100% green gate.
- **Structured Death Forensics**: Upgraded `run_games.py` to produce JSON death events and track exact causes of death across games.

---
## Fallback Forensics & Debugging Log
*Use this format for recording bugs caught in the death summary:*

**Date / Game ID**: [Date] / [GameID]
**Death Cause**: (e.g., `h2h-shorter`, `body-collision`)
**Phase**: (e.g., `LATE_1V1`)
**Fallback History (Last 10)**:
- `T41: tier=strict safe=0 chose=up score=-999.0 phase=LATE_1V1`
- `...`
**Analysis**: (Why did tier-A safety fail? What heuristic was over-weighted?)
**Resolution**: (Fix implemented and new test added to regression suite)
