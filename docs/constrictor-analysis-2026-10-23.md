# 2026-10-23: Constrictor plateau audit and correctness patch

The profile screenshot supplied by the player showed Constrictor rating **7,836 / rank 30**, Standard **8,148 / rank 87**, Duels **8,458 / rank 70**, and Royale **8,085 / rank 61**. This does **not** show a fixed ceiling. Leaderboards use daily TrueSkill updates based on real matches; local training or a code change alone cannot establish a leaderboard improvement.

## Found in the checked-in policy

* `TacticalEngine._score_move` chiefly rewarded static flood-filled connected area plus a territory map. Two candidate heads within one connected component could receive similar space scores although one forced an early trap after successive permanent-growth turns. Existing `survive_horizon` only checked for existence of **a** five-turn continuation, treating enemy bodies as stationary.
* Constrictor has no tail release, no hunger and no food. Its action evaluation needs to consider *self-avoiding* paths, not just reachability. We added `constrictor_search.route_quality`: 18-ply, width-72 beam search with permanent occupancy, hard board limits and a deadline, plus an estimated enemy-head arrival map as a soft risk feature. No claim is made that this predicts simultaneous opponents or solves the NP-hard longest-path problem.
* `run_games.py` standalone simulator discarded all prior head squares from collision occupancy, even though each prior head becomes a neck after a simultaneous move. It also subtracted hunger and hazard damage *after* a food health reset, resulting in 99 health for normal food and hazard damage for food on a hazard. Both discrepancies were corrected; eating snakes retain their tail for collision checks on that same turn.

## Evidence and limits

* Unit tests cover body occupancy, retained tails, food/hazard health order, Constrictor search boundaries/dead ends, deterministic behavior, expired deadlines, and integration with the deployed move selector.
* Run `python -m pytest -q tests/test_main.py tests/test_competitive_engine.py tests/test_strategy_search.py tests/test_constrictor_search.py tests/test_standalone_rule_parity.py test_seed8_regression.py`.
* Passing the tests only proves the checked invariants. **The game win rate and rating have not yet been measured** for this change; the standalone simulator remains approximate and Blackout-focused. Run paired, mirrored, fixed-seed **real-engine** tournaments across Standard, Duels, Royale and Constrictor against baseline code and varied opponents before claiming a stronger player or auto-promoting a neural checkpoint. Record mean win share, elimination cause, illegal moves, end-to-end p95 latency and the exact deployed commit.
* The Constrictor planner is designed to run within the existing internal 200ms budget, but actual hosted p95 must still be measured; server roundtrip and cold starts consume additional engine timeout.
* The neural champion is unchanged; the existing optional `NeuralAdvisor` remains behind safety filtering.

References: [official rules](https://docs.battlesnake.com/rules), [Constrictor map](https://docs.battlesnake.com/maps), [leaderboards](https://docs.battlesnake.com/guides/leaderboards), [useful algorithms](https://docs.battlesnake.com/guides/useful-algorithms).
