# Moses’ Staff v13 Tactical Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ruleset-aware, time-aware tactical search that improves Constrictor and Standard play without changing the production champion until benchmark gates pass.

**Architecture:** Extract ruleset classification and temporal board analysis into focused modules. The existing `TacticalEngine` consumes immutable analysis results, applies a deadline-aware two-ply search, and falls back to its current safe scorer when time is short. Every behavior is introduced through failing tests.

**Tech Stack:** Python 3.12, FastAPI, pytest, dataclasses, stdlib collections/time.

**Spec:** `docs/superpowers/specs/2026-09-17-moses-v13-elite-design.md`

## Global Constraints

- The deterministic safety shield remains authoritative over learned policies.
- Production behavior remains unchanged until fixed-seed benchmarks pass.
- Move computation must stay inside the existing 200 ms internal budget.
- Standard, Constrictor, Blackout and unknown rulesets must be classified explicitly.
- Constrictor must not use food or health incentives.
- All new production behavior starts with a failing test.
- The branch is `codex/moses-v13-elite`.

---

## File map

- Create `ruleset_profiles.py`: classify requests and expose immutable tactical/reward flags.
- Create `tactical_search.py`: temporal occupancy, territory races, chokepoints and bounded two-ply search.
- Modify `main.py`: construct an analysis object and blend its bounded tactical score into safe candidates.
- Create `tests/test_ruleset_profiles.py`: routing and profile invariants.
- Create `tests/test_temporal_tactics.py`: tail-clear, arrival, chokepoint and trap fixtures.
- Create `tests/test_v13_integration.py`: engine selection, deadline fallback and latency invariants.
- Modify `run_games.py`: expose deterministic candidate-vs-baseline tactical benchmark.
- Create `tests/test_v13_benchmark.py`: benchmark result schema and promotion refusal.
- Modify `README.md`: document v13 tactical candidate commands and non-promotion status.

### Task 1: Ruleset profiles

**Files:**
- Create: `ruleset_profiles.py`
- Test: `tests/test_ruleset_profiles.py`

**Interfaces:**
- Produces: `RulesetKind`, `RulesetProfile`, `classify_ruleset(data: dict[str, Any]) -> RulesetProfile`
- Consumes: Battlesnake request dictionaries.

- [ ] **Step 1: Write failing classification tests**

```python
from ruleset_profiles import RulesetKind, classify_ruleset

def payload(name: str, settings: dict | None = None) -> dict:
    return {"game": {"ruleset": {"name": name, "settings": settings or {}}}}

def test_constrictor_disables_food_and_health_objectives():
    profile = classify_ruleset(payload("constrictor"))
    assert profile.kind is RulesetKind.CONSTRICTOR
    assert profile.uses_food is False
    assert profile.uses_health is False
    assert profile.grows_every_turn is True

def test_blackout_is_detected_from_view_radius():
    profile = classify_ruleset(payload("standard", {"viewRadius": 5}))
    assert profile.kind is RulesetKind.BLACKOUT
    assert profile.partial_observation is True

def test_unknown_ruleset_uses_conservative_fallback():
    profile = classify_ruleset(payload("future-mode"))
    assert profile.kind is RulesetKind.FALLBACK
    assert profile.uses_health is True
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_ruleset_profiles.py`  
Expected: FAIL with `ModuleNotFoundError: ruleset_profiles`.

- [ ] **Step 3: Implement immutable profiles**

```python
class RulesetKind(str, Enum):
    STANDARD = "standard"
    CONSTRICTOR = "constrictor"
    BLACKOUT = "blackout"
    FALLBACK = "fallback"

@dataclass(frozen=True)
class RulesetProfile:
    kind: RulesetKind
    uses_food: bool
    uses_health: bool
    grows_every_turn: bool
    partial_observation: bool
    territory_weight: float
    trap_weight: float

def classify_ruleset(data: dict[str, Any]) -> RulesetProfile:
    ruleset = data.get("game", {}).get("ruleset", {})
    name = str(ruleset.get("name", "")).lower()
    settings = ruleset.get("settings") or {}
    if name == "constrictor":
        return RulesetProfile(RulesetKind.CONSTRICTOR, False, False, True, False, 1.45, 1.50)
    if "viewRadius" in settings or "blackout" in name:
        return RulesetProfile(RulesetKind.BLACKOUT, True, True, False, True, 1.10, 1.20)
    if name in {"standard", "solo", "royale", "wrapped"}:
        return RulesetProfile(RulesetKind.STANDARD, True, True, False, False, 1.00, 1.00)
    return RulesetProfile(RulesetKind.FALLBACK, True, True, False, False, 1.00, 1.15)
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_ruleset_profiles.py`  
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ruleset_profiles.py tests/test_ruleset_profiles.py
git commit -m "feat: add explicit Battlesnake ruleset profiles"
```

### Task 2: Temporal occupancy and tail-clear analysis

**Files:**
- Create: `tactical_search.py`
- Test: `tests/test_temporal_tactics.py`

**Interfaces:**
- Produces:
  - `TemporalBoard.from_payload(data, profile) -> TemporalBoard`
  - `TemporalBoard.clear_turn: dict[Point, int]`
  - `TemporalBoard.can_enter(point: Point, arrival_turn: int) -> bool`
  - `earliest_arrivals(board, start, deadline, max_depth) -> dict[Point, int]`
- Consumes: `RulesetProfile` from Task 1.

- [ ] **Step 1: Write failing temporal tests**

```python
def test_tail_cell_clears_before_head_cell():
    board = TemporalBoard.from_payload(constrictor_state(), CONSTRICTOR)
    body = [(3, 3), (3, 2), (3, 1)]
    assert board.clear_turn[body[-1]] < board.clear_turn[body[0]]

def test_arrival_may_use_cell_after_it_clears():
    board = TemporalBoard.from_payload(constrictor_state(), CONSTRICTOR)
    tail = (3, 1)
    assert board.can_enter(tail, board.clear_turn[tail]) is True
    assert board.can_enter(tail, board.clear_turn[tail] - 1) is False

def test_constrictor_growth_delays_own_tail_release():
    normal = TemporalBoard.from_payload(standard_state(), STANDARD)
    constrictor = TemporalBoard.from_payload(constrictor_state(), CONSTRICTOR)
    tail = (3, 1)
    assert constrictor.clear_turn[tail] > normal.clear_turn[tail]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_temporal_tactics.py -k "tail or arrival"`  
Expected: FAIL because `TemporalBoard` is undefined.

- [ ] **Step 3: Implement temporal board**

Store walls as permanently blocked and each body segment as the earliest relative turn at which it vacates. For body index `i` from head, use `len(body) - i`; add one turn under Constrictor’s continuous growth forecast. Treat duplicated tails as delayed by one additional turn. `can_enter` returns true only when `arrival_turn >= clear_turn[point]`.

- [ ] **Step 4: Implement bounded earliest-arrival BFS**

Use queue entries `(point, turn)`, reject arrivals after `max_depth`, call `board.can_enter(next_point, turn + 1)`, and stop immediately when `time.monotonic() >= deadline - 0.006`.

- [ ] **Step 5: Run temporal tests**

Run: `pytest -q tests/test_temporal_tactics.py -k "tail or arrival"`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tactical_search.py tests/test_temporal_tactics.py
git commit -m "feat: model time-aware body cell release"
```

### Task 3: Territory race and head danger map

**Files:**
- Modify: `tactical_search.py`
- Modify: `tests/test_temporal_tactics.py`

**Interfaces:**
- Produces:
  - `territory_score(board, our_start, enemies, our_length, deadline) -> float`
  - `dangerous_enemy_destinations(board, enemies, our_length) -> set[Point]`
- Consumes: `TemporalBoard`, `earliest_arrivals`.

- [ ] **Step 1: Write failing tests**

```python
def test_equal_or_longer_enemy_marks_shared_head_destination_dangerous():
    danger = dangerous_enemy_destinations(board, [enemy(head=(5, 5), length=8)], our_length=8)
    assert (5, 4) in danger
    assert (5, 6) in danger

def test_longer_snake_wins_equal_arrival_territory():
    score_long = territory_score(board, (2, 2), [enemy(head=(4, 2), length=5)], 7, deadline())
    score_short = territory_score(board, (2, 2), [enemy(head=(4, 2), length=7)], 5, deadline())
    assert score_long > score_short
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_temporal_tactics.py -k "enemy or territory"`  
Expected: FAIL with missing functions.

- [ ] **Step 3: Implement danger and territory**

Compute earliest arrival maps for us and each visible enemy. Award +1 for cells reached first, -1 for cells lost, +0.25 for ties won by length, and -0.5 for ties against equal/longer enemies. Normalize by the number of reachable board cells.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_temporal_tactics.py -k "enemy or territory"`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tactical_search.py tests/test_temporal_tactics.py
git commit -m "feat: add temporal territory and head danger maps"
```

### Task 4: Chokepoints and bounded two-ply trap search

**Files:**
- Modify: `tactical_search.py`
- Modify: `tests/test_temporal_tactics.py`

**Interfaces:**
- Produces:
  - `find_articulation_points(board, reachable, deadline) -> set[Point]`
  - `evaluate_two_ply(data, candidate, profile, deadline) -> SearchResult`
  - `SearchResult(score: float, forced_death: bool, completed: bool)`
- Consumes: Task 2 and Task 3 analysis.

- [ ] **Step 1: Add failing trap fixtures**

Create a U-shaped body fixture where moving into the pocket is legal this turn but every following move dies, and an open fixture with two exits.

```python
def test_two_ply_rejects_legal_one_turn_pocket():
    result = evaluate_two_ply(pocket_state(), (4, 4), CONSTRICTOR, deadline())
    assert result.completed is True
    assert result.forced_death is True

def test_open_region_is_not_forced_death():
    result = evaluate_two_ply(open_state(), (4, 4), CONSTRICTOR, deadline())
    assert result.forced_death is False
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_temporal_tactics.py -k "two_ply or articulation"`  
Expected: FAIL with missing functions.

- [ ] **Step 3: Implement articulation detection**

Use deadline-aware Tarjan DFS on the candidate’s reachable subgraph. Return partial results with `completed=False` when the deadline guard fires; incomplete analysis must never declare forced death.

- [ ] **Step 4: Implement two-ply minimax**

Enumerate our safe second moves against the union of each enemy’s legal next destinations. Mark `forced_death=True` only if every legal continuation is certainly fatal across plausible enemy responses. Otherwise return a score combining normalized territory, exit count, danger exposure and chokepoint penalty.

- [ ] **Step 5: Verify GREEN and runtime**

Run: `pytest -q tests/test_temporal_tactics.py -k "two_ply or articulation"`  
Expected: PASS in under 1 second.

- [ ] **Step 6: Commit**

```bash
git add tactical_search.py tests/test_temporal_tactics.py
git commit -m "feat: add deadline-aware two-ply trap search"
```

### Task 5: Integrate tactical analysis behind a candidate flag

**Files:**
- Modify: `main.py`
- Create: `tests/test_v13_integration.py`

**Interfaces:**
- Produces: `V13_TACTICS_ENABLED` environment-controlled candidate behavior.
- Consumes: `classify_ruleset`, `TemporalBoard`, `evaluate_two_ply`, `territory_score`.

- [ ] **Step 1: Write failing integration tests**

```python
def test_constrictor_candidate_ignores_food_location(monkeypatch):
    monkeypatch.setenv("V13_TACTICS_ENABLED", "1")
    left = move_for(constrictor_fixture(food=[{"x": 0, "y": 0}]))
    right = move_for(constrictor_fixture(food=[{"x": 10, "y": 10}]))
    assert left == right

def test_deadline_fallback_returns_legal_move(monkeypatch):
    monkeypatch.setattr(time, "monotonic", advancing_clock())
    move = move_for(dense_board_fixture())
    assert move in legal_moves(dense_board_fixture())
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_v13_integration.py`  
Expected: FAIL because candidate integration is absent.

- [ ] **Step 3: Integrate profile-aware scoring**

At the start of `get_best_move`, classify the ruleset. Preserve the existing strict safety pool. When the flag is enabled and at least 35 ms remain, score each safe candidate using two-ply search; remove only candidates whose completed search proves forced death. Add bounded territory/search scores multiplied by the profile weights. If analysis is incomplete, retain existing scoring unchanged.

- [ ] **Step 4: Keep neural adviser subordinate**

Call `NeuralAdvisor.scores` only after v13’s safe candidate pool is finalized. Neural scores cannot restore a removed forced-death move.

- [ ] **Step 5: Verify integration and regression**

Run: `pytest -q tests/test_v13_integration.py tests/test_competitive_engine.py tests/test_main.py`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_v13_integration.py
git commit -m "feat: integrate v13 tactical candidate safely"
```

### Task 6: Deterministic benchmark and promotion refusal

**Files:**
- Modify: `run_games.py`
- Create: `tests/test_v13_benchmark.py`
- Modify: `README.md`

**Interfaces:**
- Produces:
  - CLI `python run_games.py --compare-v13 --games 1200 --seed 1307`
  - JSON fields `candidate_points`, `baseline_points`, `delta`, `ci95_low`, `ci95_high`, `self_deaths`, `p95_latency_ms`, `eligible`.
- Consumes: current engine and flag-enabled candidate.

- [ ] **Step 1: Write failing benchmark schema tests**

```python
def test_promotion_requires_positive_lower_confidence_bound():
    report = build_promotion_report(candidate=[1, 1, 0], baseline=[1, 1, 0],
                                    self_death_delta=0.0, p95_latency_ms=20)
    assert report["eligible"] is False

def test_latency_over_150ms_refuses_promotion():
    report = build_promotion_report(candidate=[2] * 100, baseline=[0] * 100,
                                    self_death_delta=0.0, p95_latency_ms=151)
    assert report["eligible"] is False
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_v13_benchmark.py`  
Expected: FAIL with missing `build_promotion_report`.

- [ ] **Step 3: Implement paired bootstrap report**

Use a seeded `random.Random(1307)`, resample paired score deltas 10,000 times, and compute 2.5/97.5 percentiles. Set `eligible` only when `ci95_low > 0`, `self_death_delta <= 0.02`, and `p95_latency_ms < 150`.

- [ ] **Step 4: Add comparison CLI**

Rotate seats, reuse identical seeds for baseline and candidate, report Standard and Constrictor separately, and write `testing/v13_tactical_report.json`. The command exits 0 for a completed benchmark even when ineligible; malformed games or missing results exit nonzero.

- [ ] **Step 5: Run all tests and a smoke benchmark**

Run:
```bash
pytest -q
python run_games.py --compare-v13 --games 40 --seed 1307
```
Expected: tests PASS; report contains all schema fields and does not modify production weights.

- [ ] **Step 6: Document candidate status**

Add commands, gates and a warning that `V13_TACTICS_ENABLED` stays disabled in production until the 1,200-game suite passes.

- [ ] **Step 7: Commit**

```bash
git add run_games.py tests/test_v13_benchmark.py README.md
git commit -m "test: gate v13 tactics with paired tournament"
```

## Plan self-review

- Spec coverage in this phase: ruleset routing, temporal occupancy, head danger, territory, chokepoints, two-ply search, deadline degradation, production isolation, latency and tactical promotion gates.
- Deferred to phase 2 plan: canonical recurrent actor, game-scoped GRU state, privileged critic.
- Deferred to phase 3 plan: vectorized simulator differential gate, curriculum, historical league expansion, live replay balancing and dashboard metrics.
- Placeholder scan: no TBD/TODO or unspecified error-handling steps.
- Interface consistency: all cross-task types and functions are named in their producing task.
