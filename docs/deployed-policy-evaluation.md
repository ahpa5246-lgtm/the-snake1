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


## Partial observations and standalone regression findings

Hidden body slots retain their original indices when determining an opponent's head and tail. Compacting visible coordinates could otherwise invent a head or treat a visible neck as a vacating tail. Match phase counts known opponent records even when their geometry is hidden.

The broader seed sweep exposed a separate standalone simulator error: eating retained the old tail and then appended another copy, growing by two segments. A scripted one-meal regression reproduced length 3 becoming 5. The simulator now always removes one tail before appending the food-growth segment, matching the [official turn order](https://docs.battlesnake.com/rules). At implementation commit `7cee052f210bd676515be74bdb60188361755659`, the meal regression and the unchanged early-collision predicates across seeds 1–10 pass ([CI](https://github.com/ahpa5246-lgtm/the-snake1/actions/runs/39487856414)). The seed-9 failure under the double-growth simulator is not evidence of a defect under official rules.

The standalone runner remains approximate: it removes old head cells from collision occupancy, although those cells become necks, and subtracts hunger after resetting meal health, yielding 99. These existing differences are not repaired here. Use the Hisss experiment for real-engine comparisons; a green standalone seed sweep is a regression check, not a conformance certificate.

## Repeatability limits

Later CI runs with the same bounded training/evaluation settings returned different results:

| Implementation | Candidate W–L–D | Score | Per-run gate |
| --- | --- | --- | --- |
| `99df97b` | 7–8–1 | 0.46875 | Retain baseline |
| `19ad1cf` | 8–7–1 | 0.53125 | Retain baseline |
| `e25b67a` | 11–4–1 | 0.71875 | Eligible for manual review |

Evidence: [99df97b](https://github.com/ahpa5246-lgtm/the-snake1/actions/runs/39486127410), [19ad1cf](https://github.com/ahpa5246-lgtm/the-snake1/actions/runs/39486683713), [e25b67a](https://github.com/ahpa5246-lgtm/the-snake1/actions/runs/39487415760). All three completed real-engine experiments reported zero candidate invalid directions and deadline overruns. Each CI execution performs a fresh PPO update; these are not repeated evaluations of one identical candidate artifact. The source of the variation has not been isolated. The score gate is a per-run screen for manual review; repeated evaluation of fixed artifacts is still needed before claiming reliable strength gains. Nothing was promoted.
