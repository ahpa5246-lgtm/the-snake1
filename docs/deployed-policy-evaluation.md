# Evaluate the policy that is actually served

The v2 protocol selected the neural network's action directly. The server instead filters unsafe actions, scores tactics, and adds a bounded neural adviser. Those two players can make different decisions.

The v3 held-out protocol runs candidate and baseline checkpoints through `TacticalEngine.get_best_move` and the same `NeuralAdvisor.scores` implementation as `/move`, with the same decision budget. Each player owns an isolated frozen adviser; evaluation never swaps the global production model. Other players use the tactical engine without a neural model.

Reports retain checkpoint and model-state digests, mirrored seats, per-game results, adviser inference counts, latency observations and CPU configuration. An uneventful run in which either neural adviser never executes cannot pass the gate. No checkpoint is promoted.

The CI experiment freezes the checked-in champion, collects four real-engine training games using seed 2026, performs one PPO update, freezes the candidate, and plays sixteen held-out mirrored games. It is a bounded regression experiment with a 120-turn cap, not proof of broad playing strength. Scores from v2 and v3 must not be compared as if they measured the same policy. A failed or inconclusive experiment must retain the baseline and expose the result.

Run against your own separately frozen files:

```bash
python training/evaluate_candidate.py --baseline PRIVATE-baseline.pt --candidate PRIVATE-candidate.pt --output PRIVATE-report.json --training-seed 2026
```

## Measured failure and first controlled result (2026-10-20)

A real Hisss 15x15 observation contained three hidden enemy heads at `(-1, -1)`. The previous point parser accepted them. Food-race searches started outside the board's wall ring and explored an exterior grid: cProfile recorded about 909,000 queue pops and 2.479 seconds for one decision. A separate endpoint regression took about 0.918 seconds against the declared 200ms budget.

The repair excludes hidden coordinates from geometry and bounds internal distance searches by board dimensions and deadline. On the same fixture, CI measured about 13.16ms under cProfile, with one successful neural inference. These are fixture-specific observations from CI runners, not a universal latency guarantee.

At implementation commit `99df97bdc7cfc81409da7e49d90ef223e6974204`, four real training games and one PPO update produced a distinct frozen candidate. The sixteen held-out mirrored games returned 7 candidate wins, 8 losses and 1 draw (score 0.46875), with zero candidate invalid directions or budget overruns. The gate rejected the candidate and retained the baseline. This small experiment does not establish general playing strength or superiority of an agent framework.

Evidence: [CI experiment and profiler](https://github.com/ahpa5246-lgtm/the-snake1/actions/runs/39486127410); [pre-fix profile](https://github.com/ahpa5246-lgtm/the-snake1/actions/runs/39485875449). No checkpoint was promoted or deployed.
