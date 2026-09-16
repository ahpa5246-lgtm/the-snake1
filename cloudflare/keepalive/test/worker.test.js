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

test("status endpoint starts with zero checks", async () => {
  const response = await worker.fetch(new Request("https://keeper.example/"), {
    STATUS: new MemoryKV(),
    SNAKE_URL: "https://the-snake1.onrender.com/",
  });
  assert.equal(response.status, 200);
  const status = await response.json();
  assert.equal(status.total_checks, 0);
  assert.equal(status.successful_checks, 0);
  assert.equal(status.uptime_percentage, 0);
});
