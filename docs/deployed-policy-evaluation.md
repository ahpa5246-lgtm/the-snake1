# Evaluate the policy that is actually served

The v2 protocol selected the neural network's action directly. The server instead filters unsafe actions, scores tactics, and adds a bounded neural adviser. Those two players can make different decisions.

The v3 held-out protocol runs candidate and baseline checkpoints through `TacticalEngine.get_best_move` and the same `NeuralAdvisor.scores` implementation as `/move`, with the same decision budget. Each player owns an isolated frozen adviser; evaluation never swaps the global production model. Other players use the tactical engine without a neural model.

Reports retain checkpoint and model-state digests, mirrored seats, per-game results, adviser inference counts, latency observations and CPU configuration. An uneventful run in which either neural adviser never executes cannot pass the gate. No checkpoint is promoted.

The CI experiment freezes the checked-in champion, collects four real-engine training games using seed 2026, performs one PPO update, freezes the candidate, and plays sixteen held-out mirrored games. It is a bounded regression experiment with a 120-turn cap, not proof of broad playing strength. Scores from v2 and v3 must not be compared as if they measured the same policy. A failed or inconclusive experiment must retain the baseline and expose the result.

Run against your own separately frozen files:

```bash
python training/evaluate_candidate.py --baseline PRIVATE-baseline.pt --candidate PRIVATE-candidate.pt --output PRIVATE-report.json --training-seed 2026
```
