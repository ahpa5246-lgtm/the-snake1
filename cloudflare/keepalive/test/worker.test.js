import assert from "node:assert/strict";
import test from "node:test";
import worker from "../src/index.js";

class MemoryKV {
  constructor() { this.values = new Map(); }
  async get(key, type) {
    const value = this.values.get(key);
    return type === "json" && value ? JSON.parse(value) : value ?? null;
  }
  async put(key, value) { this.values.set(key, value); }
}

function completedGame(id, won, endedAt = "2026-09-16T12:00:00.000Z") {
  return {
    name: `${id}.jsonl`,
    events: [
      { type: "start", game_id: id },
      { type: "move", game_id: id, turn: 1, move: "up" },
      { type: "end", game_id: id, turn: 12, won, final_length: 5, ended_at_utc: endedAt },
    ],
  };
}

test("status endpoint starts with zero checks", async () => {
  const response = await worker.fetch(new Request("https://keeper.example/api/status"), {
    STATUS: new MemoryKV(),
    SNAKE_URL: "https://the-snake1.onrender.com/",
  });
  assert.equal(response.status, 200);
  const status = await response.json();
  assert.equal(status.uptime.total_checks, 0);
  assert.equal(status.uptime.successful_checks, 0);
  assert.equal(status.uptime.uptime_percentage, 0);
});

test("scheduled checks persist each completed replay once", async () => {
  const kv = new MemoryKV();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.includes("api.github.com")) {
      return Response.json({ workflow_runs: [{ id: 7, status: "completed", conclusion: "success", html_url: "https://github.test/run/7", run_started_at: "2026-09-16T11:00:00Z", updated_at: "2026-09-16T11:10:00Z" }] });
    }
    if (url.endsWith("/learning/replays")) {
      return Response.json({ games: [completedGame("g-1", true), completedGame("g-2", false)] });
    }
    return new Response("ok");
  };
  try {
    const env = { STATUS: kv, SNAKE_URL: "https://snake.example/" };
    await worker.scheduled({}, env, { waitUntil: (promise) => promise });
    await worker.scheduled({}, env, { waitUntil: (promise) => promise });
    const response = await worker.fetch(new Request("https://keeper.example/api/status"), env);
    const status = await response.json();
    assert.equal(status.matches.total, 2);
    assert.equal(status.matches.wins, 1);
    assert.equal(status.matches.losses, 1);
    assert.equal(status.matches.daily["2026-09-16"].total, 2);
    assert.equal(status.training.status, "completed");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("replay export and Arabic dashboard are public", async () => {
  const kv = new MemoryKV();
  await kv.put("moses-staff-replay:g-9", JSON.stringify(completedGame("g-9", true)));
  await kv.put("moses-staff-replay-index", JSON.stringify(["g-9"]));
  const env = { STATUS: kv, SNAKE_URL: "https://snake.example/" };

  const replayResponse = await worker.fetch(new Request("https://keeper.example/api/replays"), env);
  const replayPayload = await replayResponse.json();
  assert.equal(replayPayload.games.length, 1);

  const dashboardResponse = await worker.fetch(new Request("https://keeper.example/"), env);
  assert.match(dashboardResponse.headers.get("content-type"), /text\/html/);
  assert.match(await dashboardResponse.text(), /لوحة موسى/);
});
