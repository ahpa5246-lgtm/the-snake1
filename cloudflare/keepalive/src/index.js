const STATUS_KEY = "moses-staff-uptime";
const MATCH_KEY = "moses-staff-matches";
const REPLAY_INDEX_KEY = "moses-staff-replay-index";
const REPLAY_PREFIX = "moses-staff-replay:";
const TRAINING_KEY = "moses-staff-training";
const GITHUB_RUNS_URL = "https://api.github.com/repos/ahpa5246-lgtm/the-snake1/actions/workflows/continual-learning.yml/runs?per_page=1";

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

function emptyMatches() {
  return {
    total: 0, wins: 0, losses: 0, win_rate: 0,
    total_turns: 0, average_turns: 0,
    first_match_utc: null, last_match_utc: null,
    last_sync_utc: null, last_sync_error: null,
    daily: {}, recent: [],
  };
}

async function syncReplays(env) {
  const now = new Date().toISOString();
  const matches = (await env.STATUS.get(MATCH_KEY, "json")) || emptyMatches();
  const index = (await env.STATUS.get(REPLAY_INDEX_KEY, "json")) || [];
  const known = new Set(index);
  let imported = 0;
  try {
    const base = env.SNAKE_URL.replace(/\/$/, "");
    const response = await fetch(`${base}/learning/replays`, {
      headers: { "User-Agent": "Moses-Staff-Cloudflare-Collector/1.0" },
      signal: AbortSignal.timeout(25000),
    });
    if (!response.ok) throw new Error(`Replay export HTTP ${response.status}`);
    const payload = await response.json();
    for (const game of Array.isArray(payload.games) ? payload.games : []) {
      const events = Array.isArray(game?.events) ? game.events : [];
      const end = events.at(-1);
      if (!end || end.type !== "end") continue;
      const gameId = String(end.game_id || game.name || "").replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 128);
      if (!gameId || known.has(gameId)) continue;

      const endedAt = end.ended_at_utc || now;
      const day = endedAt.slice(0, 10);
      const won = Boolean(end.won);
      const turns = Math.max(0, Number(end.turn) || 0);
      await env.STATUS.put(`${REPLAY_PREFIX}${gameId}`, JSON.stringify({ name: game.name || `${gameId}.jsonl`, events }));
      index.push(gameId);
      known.add(gameId);
      imported += 1;
      matches.total += 1;
      matches.wins += won ? 1 : 0;
      matches.losses += won ? 0 : 1;
      matches.total_turns += turns;
      matches.first_match_utc ||= endedAt;
      matches.last_match_utc = endedAt;
      const daily = matches.daily[day] || { total: 0, wins: 0, losses: 0, win_rate: 0 };
      daily.total += 1;
      daily.wins += won ? 1 : 0;
      daily.losses += won ? 0 : 1;
      daily.win_rate = Number(((daily.wins / daily.total) * 100).toFixed(1));
      matches.daily[day] = daily;
      matches.recent.unshift({ game_id: gameId, won, turns, final_length: Number(end.final_length) || 0, ended_at_utc: endedAt });
    }
    matches.recent = matches.recent.slice(0, 50);
    const days = Object.keys(matches.daily).sort().slice(-90);
    matches.daily = Object.fromEntries(days.map((day) => [day, matches.daily[day]]));
    matches.win_rate = matches.total ? Number(((matches.wins / matches.total) * 100).toFixed(1)) : 0;
    matches.average_turns = matches.total ? Number((matches.total_turns / matches.total).toFixed(1)) : 0;
    if (imported || matches.last_sync_error) {
      matches.last_sync_utc = now;
      matches.last_sync_error = null;
      await env.STATUS.put(REPLAY_INDEX_KEY, JSON.stringify(index.slice(-500)));
      await env.STATUS.put(MATCH_KEY, JSON.stringify(matches));
    }
  } catch (error) {
    const message = String(error?.message || error).slice(0, 300);
    if (matches.last_sync_error !== message) {
      matches.last_sync_utc = now;
      matches.last_sync_error = message;
      await env.STATUS.put(MATCH_KEY, JSON.stringify(matches));
    }
  }
  return matches;
}

async function refreshTraining(env) {
  try {
    const response = await fetch(GITHUB_RUNS_URL, {
      headers: { Accept: "application/vnd.github+json", "User-Agent": "Moses-Staff-Dashboard/1.0" },
      signal: AbortSignal.timeout(15000),
    });
    if (!response.ok) throw new Error(`GitHub HTTP ${response.status}`);
    const run = (await response.json()).workflow_runs?.[0];
    if (!run) return null;
    const training = {
      run_id: run.id, status: run.status, conclusion: run.conclusion,
      started_at_utc: run.run_started_at || run.created_at,
      updated_at_utc: run.updated_at, url: run.html_url,
      is_training_now: ["queued", "in_progress", "waiting", "pending"].includes(run.status),
    };
    const previous = await env.STATUS.get(TRAINING_KEY, "json");
    if (JSON.stringify(previous) !== JSON.stringify(training)) {
      await env.STATUS.put(TRAINING_KEY, JSON.stringify(training));
    }
    return training;
  } catch (error) {
    const previous = (await env.STATUS.get(TRAINING_KEY, "json")) || {};
    previous.refresh_error = String(error?.message || error).slice(0, 300);
    await env.STATUS.put(TRAINING_KEY, JSON.stringify(previous));
    return previous;
  }
}

async function combinedStatus(env) {
  return {
    uptime: await readStatus(env),
    matches: (await env.STATUS.get(MATCH_KEY, "json")) || emptyMatches(),
    training: (await env.STATUS.get(TRAINING_KEY, "json")) || null,
  };
}

async function exportReplays(env) {
  const index = (await env.STATUS.get(REPLAY_INDEX_KEY, "json")) || [];
  const selected = index.slice(-250);
  const games = (await Promise.all(selected.map((id) => env.STATUS.get(`${REPLAY_PREFIX}${id}`, "json")))).filter(Boolean);
  return { generated_at_utc: new Date().toISOString(), games, export_limit: 250 };
}

function dashboardHtml() {
  return `<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>لوحة موسى</title><style>
  :root{color-scheme:dark;--bg:#07131f;--card:#102538;--cyan:#50dfcf;--gold:#f1bd5b;--muted:#9db1c3}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#12324b,var(--bg) 45%);color:#f7fbff;font-family:system-ui,sans-serif;min-height:100vh}.wrap{width:min(1100px,92%);margin:auto;padding:52px 0}.eyebrow{color:var(--cyan);font-weight:800}.hero{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:28px}h1{font-size:clamp(2rem,6vw,4.5rem);margin:.1em 0}.sub{color:var(--muted)}.live{padding:9px 14px;border:1px solid #2b5067;border-radius:999px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}.card{background:linear-gradient(145deg,#122c42,#0d2031);border:1px solid #1f425b;border-radius:18px;padding:20px;box-shadow:0 18px 50px #0005}.num{font-size:2.25rem;font-weight:900;color:var(--gold)}.label{color:var(--muted);margin-top:5px}.wide{grid-column:1/-1}.bar{height:12px;background:#07131f;border-radius:99px;overflow:hidden;margin-top:12px}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--gold));width:0}table{width:100%;border-collapse:collapse;margin-top:10px}td,th{padding:12px;text-align:right;border-bottom:1px solid #23435a}.win{color:var(--cyan)}.loss{color:#ff8e93}a{color:var(--cyan)}@media(max-width:600px){.hero{display:block}.live{display:inline-block}.wrap{padding-top:28px}}
  </style></head><body><main class="wrap"><section class="hero"><div><div class="eyebrow">MOSES' STAFF • BATTLESNAKE</div><h1>لوحة موسى</h1><div class="sub" id="updated">تحميل أحدث البيانات…</div></div><div class="live" id="training">فحص التدريب…</div></section><section class="grid"><article class="card"><div class="num" id="today">0</div><div class="label">مباريات اليوم</div></article><article class="card"><div class="num" id="wins">0</div><div class="label">إجمالي الفوز</div></article><article class="card"><div class="num" id="losses">0</div><div class="label">إجمالي الخسارة</div></article><article class="card"><div class="num" id="rate">0%</div><div class="label">نسبة الفوز</div></article><article class="card wide"><b>التقدم العام</b><div class="bar"><div class="fill" id="bar"></div></div><p class="sub" id="summary"></p></article><article class="card wide"><b>آخر المباريات</b><table><thead><tr><th>النتيجة</th><th>الحركات</th><th>الوقت</th><th>معرّف المباراة</th></tr></thead><tbody id="recent"></tbody></table></article></section></main><script>
  const f=d=>d?new Date(d).toLocaleString('ar-IQ',{timeZone:'Asia/Baghdad'}):'—'; async function load(){const s=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());const m=s.matches;const day=new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Baghdad',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());document.querySelector('#today').textContent=m.daily[day]?.total||0;document.querySelector('#wins').textContent=m.wins;document.querySelector('#losses').textContent=m.losses;document.querySelector('#rate').textContent=m.win_rate+'%';document.querySelector('#bar').style.width=m.win_rate+'%';document.querySelector('#updated').textContent='آخر مباراة: '+f(m.last_match_utc)+' • آخر مزامنة: '+f(m.last_sync_utc);document.querySelector('#summary').textContent='لعب '+m.total+' مباراة بمتوسط '+m.average_turns+' حركة.';const t=s.training;document.querySelector('#training').innerHTML=t?(t.is_training_now?'يتدرّب الآن':'آخر تدريب: '+(t.conclusion||t.status)+' — <a href="'+t.url+'" target="_blank">التفاصيل</a>'):'لم تُسجّل دورة تدريب بعد';document.querySelector('#recent').innerHTML=m.recent.map(x=>'<tr><td class="'+(x.won?'win':'loss')+'">'+(x.won?'فوز':'خسارة')+'</td><td>'+x.turns+'</td><td>'+f(x.ended_at_utc)+'</td><td>'+x.game_id+'</td></tr>').join('')||'<tr><td colspan="4">لا توجد مباريات مسجلة بعد</td></tr>'}load();setInterval(load,60000);
  </script></body></html>`;
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
    return ctx.waitUntil(Promise.all([checkSnake(env), syncReplays(env), refreshTraining(env)]));
  },

  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/check") {
      await Promise.all([checkSnake(env), syncReplays(env), refreshTraining(env)]);
      return Response.json(await combinedStatus(env), {
        headers: { "Cache-Control": "no-store" },
      });
    }
    if (url.pathname === "/api/status") return Response.json(await combinedStatus(env), {
      headers: { "Cache-Control": "no-store" },
    });
    if (url.pathname === "/api/replays") return Response.json(await exportReplays(env), {
      headers: { "Cache-Control": "no-store" },
    });
    if (url.pathname === "/") return new Response(dashboardHtml(), {
      headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" },
    });
    return new Response("Not found", { status: 404 });
  },
};
