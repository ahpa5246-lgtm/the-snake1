const STATUS_KEY = "moses-staff-uptime";

function emptyStatus() {
  return {
    monitoring_started_utc: null,
    last_check_utc: null,
    last_success_utc: null,
    total_checks: 0,
    successful_checks: 0,
    failed_checks: 0,
    consecutive_failures: 0,
    uptime_percentage: 0,
    last_http_status: null,
    last_latency_ms: null,
    last_error: null,
  };
}

async function readStatus(env) {
  return (await env.STATUS.get(STATUS_KEY, "json")) || emptyStatus();
}

async function checkSnake(env) {
  const now = new Date().toISOString();
  const status = await readStatus(env);
  status.monitoring_started_utc ||= now;
  status.last_check_utc = now;
  status.total_checks += 1;
  const started = Date.now();

  try {
    const response = await fetch(env.SNAKE_URL, {
      headers: { "User-Agent": "Moses-Staff-Cloudflare-Keeper/1.0" },
      signal: AbortSignal.timeout(25000),
      redirect: "follow",
    });
    status.last_http_status = response.status;
    status.last_latency_ms = Date.now() - started;
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    status.successful_checks += 1;
    status.consecutive_failures = 0;
    status.last_success_utc = now;
    status.last_error = null;
  } catch (error) {
    status.failed_checks += 1;
    status.consecutive_failures += 1;
    status.last_latency_ms = Date.now() - started;
    status.last_error = String(error && error.message ? error.message : error).slice(0, 300);
  }

  status.uptime_percentage = Number(
    ((status.successful_checks / Math.max(1, status.total_checks)) * 100).toFixed(3),
  );
  await env.STATUS.put(STATUS_KEY, JSON.stringify(status));
  return status;
}

export default {
  async scheduled(_controller, env, ctx) {
    ctx.waitUntil(checkSnake(env));
  },

  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/check") {
      return Response.json(await checkSnake(env), {
        headers: { "Cache-Control": "no-store" },
      });
    }
    return Response.json(await readStatus(env), {
      headers: { "Cache-Control": "no-store" },
    });
  },
};
