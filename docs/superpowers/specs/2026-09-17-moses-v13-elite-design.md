# Moses’ Staff v13 Elite — Design Specification

**Date:** 2026-09-17  
**Status:** Approved direction; implementation pending plan  
**Branch:** `codex/moses-v13-elite`

## Goal

Build a substantially stronger Battlesnake agent inspired by the published Snakity Blackout 2026 methods while preserving Moses’ Staff’s deterministic safety layer, free-hosting constraints, live replay learning, and compatibility with Standard, Constrictor, and partial-observation rulesets.

The upgrade must earn promotion through reproducible head-to-head evaluation. No claim of improvement is made from architecture alone.

## Success criteria

A candidate may replace the current champion only when all gates pass:

1. At least 1,200 deterministic evaluation games across fixed unseen seeds.
2. Separate suites for Standard 11x11, Constrictor 11x11, and Blackout-like 15x15.
3. Overall score improvement with a 95% bootstrap confidence interval whose lower bound is above zero.
4. No regression greater than 2 percentage points in immediate/self-collision deaths.
5. p95 move latency below 150 ms on a CPU representative of Render.
6. Zero invalid moves, uncaught inference errors, or checkpoint corruption.
7. Constrictor is evaluated separately and cannot be hidden by stronger Standard results.

## Architecture

### 1. Ruleset router

Introduce an explicit ruleset profile derived from the request:

- `standard`: health, food and hazards matter.
- `constrictor`: growth every turn, no hunger objective, territory and tail-clear timing dominate.
- `blackout`: partial observation, decaying memory and uncertainty penalties.
- `fallback`: conservative standard-compatible behavior for unknown rulesets.

Each profile owns its reward terms, tactical weights, observation channels, evaluation suite, and promotion score. A model trained for one profile cannot silently be promoted for all profiles.

### 2. Deterministic safety shield

The existing safety-first engine remains authoritative. Expand it with:

- Exact vacating-tail time for every occupied cell.
- Simultaneous enemy next-head reachability.
- Head-to-head outcome map with pessimistic uncertainty.
- Two-ply adversarial trap detection under the remaining time budget.
- Articulation/chokepoint detection to avoid entering regions an opponent can seal.
- Time-aware flood fill that admits cells only when they clear before arrival.
- Territory race map based on earliest arrival and snake length.
- Constrictor-specific survival capacity and enclosure pressure.
- Deadline-aware degradation: two-ply search → time-aware territory → current safe fallback.

The learned policy only ranks moves that survive the safety shield.

### 3. Canonical recurrent policy

Replace the current global-average CNN adviser with a compact recurrent actor:

- Head-centered 13x13 crop.
- Board rotated so current heading always points forward.
- Relative actions: forward, left, right; reverse is structurally impossible.
- Spatial channels:
  - unknown/unseen,
  - known empty,
  - own body,
  - opponent bodies,
  - food,
  - hazards/off-board,
  - memory age,
  - head-to-head advantage,
  - ticks until each cell clears,
  - earliest-arrival territory advantage,
  - legal/safety mask,
  - ruleset identity planes.
- Normalized scalar inputs: length, health, turn, board dimensions, opponents alive, and phase.
- Lightweight convolutional encoder with spatially preserving flatten/attention pooling.
- GRU memory of 128 dimensions for concurrent-game state.
- Policy head for three relative actions.
- Value head for inference diagnostics.

GRU state is keyed by game ID, reset on `/start` and removed on `/end`. State storage is bounded and expires stale games.

### 4. Privileged training critic

Training uses a separate critic that may see the full simulator state, absolute opponent bodies, all food, and exact tail-clear times. The live actor receives only legal information. This asymmetry improves the learning signal without leaking privileged information during play.

### 5. Multi-ruleset curriculum

Training proceeds in stages:

1. Legal movement and survival against safe-random opponents.
2. Tactical baseline opponents.
3. Historical checkpoint pool.
4. Diverse scripted opponents: hungry, space-seeking, aggressive, wall-following, and trap-setting.
5. Mixed self-play with deliberately sampled hard and underrepresented opponents.
6. Final replay fine-tuning from real Battlesnake matches using conservative updates.

Opponent snapshots are never discarded solely because they are old. Sampling is diversity-aware rather than aggressively prioritizing only near-50% opponents.

### 6. Reward design

Rewards are potential-based where possible so shaping does not change the optimal terminal objective.

Common terminal component:

- first place / sole survivor,
- placement score for multiplayer games,
- elimination loss,
- draw handling matching the ruleset.

Standard adds health survival, food urgency, hazard cost and safe length advantage.

Constrictor removes food/health incentives and adds:

- reachable future capacity,
- territory ownership,
- opponent enclosure progress,
- avoiding articulation-point traps,
- survival time only as a small shaping term.

Blackout adds information gain, uncertainty-aware safety, and memory consistency.

Truncated games bootstrap from the critic value rather than pretending the outcome is zero.

### 7. Faster reproducible simulator path

Keep HISSS as the correctness oracle. Add an optional vectorized batch simulator for training throughput only. Before it can train production candidates, differential tests must compare thousands of transitions against HISSS across seeds and edge cases. Any mismatch disables the fast path.

Free GitHub Actions remains the default trainer. Work is divided into resumable bounded jobs with atomic checkpoints and immutable manifests. No paid GPU is required, though the design can use one later.

### 8. Champion league and promotion

Every snapshot enters a league containing:

- current production champion,
- recent snapshots,
- strategically diverse historical snapshots,
- deterministic tactical agent,
- scripted baselines.

Evaluation uses fixed unseen seeds, rotated seat positions and paired matchups. Select by ruleset-weighted points, not newest timestamp. Report win/place rates, confidence intervals, self-deaths, timeouts and latency.

Promotion is atomic: write a temporary checkpoint, verify hash/schema, then rename. Rollback retains the previous champion.

### 9. Live learning safety

Real matches continue to be recorded and deduplicated. Live data may fine-tune a candidate but cannot directly overwrite production. Replay training is weighted to prevent a small run of similar Constrictor games from erasing general competence.

The dashboard will expose:

- production model/version,
- last training and promotion decision,
- games used by ruleset,
- candidate vs champion score,
- confidence interval,
- rejection reason,
- latency and safety regressions.

## Implementation boundaries

Likely new modules:

- `ruleset_profiles.py`
- `tactical_search.py`
- `elite_policy.py`
- `training/elite_selfplay.py`
- `training/champion_league.py`
- `training/differential_sim_test.py`

Existing integration points:

- `main.py`
- `training/live_finetune.py`
- `training/evaluate_candidate.py`
- GitHub Actions training workflows
- dashboard learning-status payload

The current champion and production route remain intact until every promotion gate passes.

## Testing strategy

Tests are written before implementation.

- Unit tests for rotation transforms and relative/absolute action conversion.
- Tail-clear and simultaneous-collision fixtures.
- Ruleset-specific reward tests proving Constrictor ignores food/health.
- Recurrent-state isolation tests for concurrent games.
- Deadline and fallback tests.
- Checkpoint atomicity and schema tests.
- Differential simulator tests against HISSS.
- Fixed-seed regression tournaments.
- CPU latency benchmark with recorded p50/p95/p99.

## Risks and controls

- **Insufficient free compute:** prioritize tactical gains and compact models; use resumable training.
- **Overfitting Constrictor:** independent ruleset suites and replay balancing.
- **Simulator mismatch:** HISSS differential gate.
- **Recurrent state leakage:** game-scoped state with lifecycle and TTL tests.
- **Stronger model but slower server:** CPU latency gate and deterministic fallback.
- **Catastrophic promotion:** confidence-based league gate and atomic rollback.
