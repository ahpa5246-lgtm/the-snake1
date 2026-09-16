# Live Snake Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist live match history and expose a public Arabic dashboard with match and training status.

**Architecture:** The existing Cloudflare Worker polls Render every five minutes, stores deduplicated replays and aggregate statistics in KV, and refreshes the latest public GitHub Actions run. The learner consumes the Worker's durable replay export.

**Tech Stack:** Cloudflare Workers, Workers KV, JavaScript, Node test runner, GitHub Actions, Python learner.

**Spec:** `docs/superpowers/specs/2026-09-16-live-dashboard-design.md`

## Global Constraints

- Keep the existing five-minute keepalive schedule.
- Do not train or deploy a candidate without the existing 32-game promotion gate.
- Store no secrets or personal data in the public dashboard.

---

### Task 1: Persistent match ingestion

**Files:** Modify `cloudflare/keepalive/src/index.js`; modify `cloudflare/keepalive/test/worker.test.js`.

- [ ] Write failing tests for replay deduplication and daily statistics.
- [ ] Run `npm test` and confirm feature assertions fail.
- [ ] Implement bounded KV replay storage and aggregation.
- [ ] Run `npm test` and confirm all tests pass.

### Task 2: Public dashboard and training state

**Files:** Modify `cloudflare/keepalive/src/index.js`; modify `cloudflare/keepalive/test/worker.test.js`.

- [ ] Write failing tests for dashboard HTML, JSON status, and GitHub run state.
- [ ] Run `npm test` and confirm feature assertions fail.
- [ ] Implement the endpoints and scheduled refresh.
- [ ] Run `npm test` and confirm all tests pass.

### Task 3: Durable learner source and documentation

**Files:** Modify `.github/workflows/continual-learning.yml`; modify `cloudflare/keepalive/README.md`.

- [ ] Point replay import at `/api/replays`.
- [ ] Document dashboard endpoints and deployment command.
- [ ] Run Worker tests and YAML syntax validation.
