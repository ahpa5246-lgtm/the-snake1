const STATUS_KEY = "moses-staff-uptime";
const MATCH_KEY = "moses-staff-matches";
const REPLAY_INDEX_KEY = "moses-staff-replay-index";
const REPLAY_PREFIX = "moses-staff-replay:";
const TRAINING_KEY = "moses-staff-training";
const GITHUB_RUNS_URL = "https://api.github.com/repos/ahpa5246-lgtm/the-snake1/actions/workflows/continual-learning.yml/runs?per_page=1";
const GITHUB_ACTIONS_URL = "https://github.com/ahpa5246-lgtm/the-snake1/actions/workflows/continual-learning.yml";
const GITHUB_BADGE_URL = `${GITHUB_ACTIONS_URL}/badge.svg?branch=main`;

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
    schema_version: 2,
    total: 0, wins: 0, losses: 0, win_rate: 0,
    total_turns: 0, average_turns: 0,
    first_match_utc: null, last_match_utc: null,
    last_sync_utc: null, last_sync_error: null,
    activity: { currently_playing: false, active_games: 0, last_activity_utc: null },
    daily: {}, recent: [],
  };
}

function baghdadDay(value) {
  const parts = new Intl.DateTimeFormat("en", {
    timeZone: "Asia/Baghdad", year: "numeric", month: "2-digit", day: "2-digit",
  }).formatToParts(new Date(value));
  const part = (type) => parts.find((item) => item.type === type)?.value;
  return `${part("year")}-${part("month")}-${part("day")}`;
}

function shiftDay(day, offset) {
  const date = new Date(`${day}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() + offset);
  return date.toISOString().slice(0, 10);
}

function addGame(matches, game, now) {
  const events = Array.isArray(game?.events) ? game.events : [];
  const end = events.at(-1);
  if (!end || end.type !== "end") return null;
  const gameId = String(end.game_id || game.name || "").replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 128);
  if (!gameId) return null;
  const endedAt = end.ended_at_utc || now;
  const day = baghdadDay(endedAt);
  const won = Boolean(end.won);
  const turns = Math.max(0, Number(end.turn) || 0);
  matches.total += 1;
  matches.wins += won ? 1 : 0;
  matches.losses += won ? 0 : 1;
  matches.total_turns += turns;
  if (!matches.first_match_utc || new Date(endedAt) < new Date(matches.first_match_utc)) matches.first_match_utc = endedAt;
  if (!matches.last_match_utc || new Date(endedAt) > new Date(matches.last_match_utc)) matches.last_match_utc = endedAt;
  const daily = matches.daily[day] || { total: 0, wins: 0, losses: 0, turns: 0, win_rate: 0, average_turns: 0 };
  daily.total += 1;
  daily.wins += won ? 1 : 0;
  daily.losses += won ? 0 : 1;
  daily.turns += turns;
  daily.win_rate = Number(((daily.wins / daily.total) * 100).toFixed(1));
  daily.average_turns = Number((daily.turns / daily.total).toFixed(1));
  matches.daily[day] = daily;
  matches.recent.push({ game_id: gameId, won, turns, final_length: Number(end.final_length) || 0, ended_at_utc: endedAt });
  return gameId;
}

function finishMatches(matches) {
  matches.recent.sort((a, b) => new Date(b.ended_at_utc) - new Date(a.ended_at_utc));
  matches.recent = matches.recent.slice(0, 50);
  const days = Object.keys(matches.daily).sort().slice(-90);
  matches.daily = Object.fromEntries(days.map((day) => [day, matches.daily[day]]));
  matches.win_rate = matches.total ? Number(((matches.wins / matches.total) * 100).toFixed(1)) : 0;
  matches.average_turns = matches.total ? Number((matches.total_turns / matches.total).toFixed(1)) : 0;
  return matches;
}

async function migrateMatches(env, index, stored, now) {
  if (stored?.schema_version === 2) return stored;
  const migrated = emptyMatches();
  const games = (await Promise.all(index.map((id) => env.STATUS.get(`${REPLAY_PREFIX}${id}`, "json")))).filter(Boolean);
  for (const game of games) addGame(migrated, game, now);
  migrated.last_sync_utc = stored?.last_sync_utc || null;
  migrated.last_sync_error = stored?.last_sync_error || null;
  return finishMatches(migrated);
}

async function syncReplays(env) {
  const now = new Date().toISOString();
  const index = (await env.STATUS.get(REPLAY_INDEX_KEY, "json")) || [];
  const stored = await env.STATUS.get(MATCH_KEY, "json");
  const matches = await migrateMatches(env, index, stored, now);
  const known = new Set(index);
  let imported = 0;
  let changed = stored?.schema_version !== 2;
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
      await env.STATUS.put(`${REPLAY_PREFIX}${gameId}`, JSON.stringify({ name: game.name || `${gameId}.jsonl`, events }));
      index.push(gameId);
      known.add(gameId);
      imported += 1;
      addGame(matches, game, now);
    }
    finishMatches(matches);
    const nextActivity = payload.activity || matches.activity;
    if (JSON.stringify(nextActivity) !== JSON.stringify(matches.activity)) changed = true;
    matches.activity = nextActivity;
    if (imported || matches.last_sync_error || changed) {
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
    const apiError = String(error?.message || error).slice(0, 300);
    try {
      const badge = await fetch(GITHUB_BADGE_URL, { signal: AbortSignal.timeout(15000) });
      if (!badge.ok) throw new Error(`GitHub badge HTTP ${badge.status}`);
      const svg = await badge.text();
      const label = svg.match(/Continual Learning\s*-\s*([^<]+)/i)?.[1]?.trim().toLowerCase() || "unknown";
      const conclusion = label.includes("pass") || label.includes("success") ? "success" : label.includes("fail") ? "failure" : label;
      const training = { status: "completed", conclusion, is_training_now: false, url: GITHUB_ACTIONS_URL, source: "status_badge", api_error: apiError };
      const previous = await env.STATUS.get(TRAINING_KEY, "json");
      if (JSON.stringify(previous) !== JSON.stringify(training)) await env.STATUS.put(TRAINING_KEY, JSON.stringify(training));
      return training;
    } catch (badgeError) {
      const training = { status: "unavailable", conclusion: null, is_training_now: null, url: GITHUB_ACTIONS_URL, refresh_error: `${apiError}; ${badgeError.message}` };
      const previous = await env.STATUS.get(TRAINING_KEY, "json");
      if (JSON.stringify(previous) !== JSON.stringify(training)) await env.STATUS.put(TRAINING_KEY, JSON.stringify(training));
      return training;
    }
  }
}

function buildAnalytics(matches, uptime, now = new Date()) {
  const todayKey = baghdadDay(now);
  const yesterdayKey = shiftDay(todayKey, -1);
  const zero = { total: 0, wins: 0, losses: 0, win_rate: 0, average_turns: 0 };
  const today = matches.daily[todayKey] || zero;
  const yesterday = matches.daily[yesterdayKey] || zero;
  const lastTen = matches.recent.slice(0, 10);
  const lastTenWins = lastTen.filter((game) => game.won).length;
  let streak = 0;
  if (matches.recent.length) {
    const result = matches.recent[0].won;
    for (const game of matches.recent) { if (game.won !== result) break; streak += 1; }
    streak *= result ? 1 : -1;
  }
  const lastSuccessAge = uptime.last_success_utc ? now - new Date(uptime.last_success_utc) : Infinity;
  return {
    timezone: "Asia/Baghdad", today_key: todayKey, yesterday_key: yesterdayKey,
    today, yesterday,
    win_rate_change: Number((today.win_rate - yesterday.win_rate).toFixed(1)),
    last_ten_win_rate: lastTen.length ? Number(((lastTenWins / lastTen.length) * 100).toFixed(1)) : 0,
    current_streak: streak,
    playing_now: Boolean(matches.activity?.currently_playing),
    active_games: Number(matches.activity?.active_games || 0),
    cloudflare_healthy: uptime.consecutive_failures === 0,
    render_healthy: uptime.last_http_status === 200 && lastSuccessAge < 10 * 60 * 1000,
    collector_healthy: !matches.last_sync_error,
    history: Object.entries(matches.daily).sort(([a], [b]) => b.localeCompare(a)).map(([date, stats]) => ({ date, ...stats })),
  };
}

async function combinedStatus(env) {
  const uptime = await readStatus(env);
  const matches = (await env.STATUS.get(MATCH_KEY, "json")) || emptyMatches();
  return { uptime, matches, analytics: buildAnalytics(matches, uptime), training: (await env.STATUS.get(TRAINING_KEY, "json")) || null };
}

async function exportReplays(env) {
  const index = (await env.STATUS.get(REPLAY_INDEX_KEY, "json")) || [];
  const selected = index.slice(-250);
  const games = (await Promise.all(selected.map((id) => env.STATUS.get(`${REPLAY_PREFIX}${id}`, "json")))).filter(Boolean);
  return { generated_at_utc: new Date().toISOString(), games, export_limit: 250 };
}

function dashboardHtml() {
  return `<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>غرفة عمليات موسى</title><style>
  :root{color-scheme:dark;--canvas:#06121c;--surface:#0b1c29;--raised:#102838;--edge:#1e4357;--text:#f2f8fb;--muted:#8da7b8;--cyan:#53e0cf;--amber:#ffbe55;--red:#ff7580;--green:#52d69a;--focus:#fff;--r:16px}*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--text);font-family:Tahoma,"Segoe UI",sans-serif;min-height:100vh}.shell{width:min(1380px,94%);margin:auto;padding:28px 0 60px}.mast{display:grid;grid-template-columns:1fr auto;gap:28px;align-items:end;padding:20px 0 26px;border-bottom:1px solid var(--edge)}.brand{display:flex;gap:16px;align-items:center}.mark{width:58px;height:58px;border:2px solid var(--cyan);display:grid;place-items:center;font:bold 30px monospace;clip-path:polygon(0 0,82% 0,100% 18%,100% 100%,18% 100%,0 82%)}.kicker{font:700 12px monospace;letter-spacing:.14em;color:var(--cyan)}h1{font-size:clamp(2rem,5vw,4.4rem);line-height:1;margin:.18em 0}.muted{color:var(--muted)}.live-banner{min-width:260px;padding:18px 20px;background:var(--surface);border:1px solid var(--edge);border-right:4px solid var(--amber)}.live-banner strong{display:block;font-size:1.15rem;margin-bottom:5px}.status-line{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--edge);margin:18px 0}.status{background:var(--surface);padding:14px 16px;display:flex;align-items:center;gap:9px}.dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}.ok .dot{background:var(--green);box-shadow:0 0 0 4px #52d69a18}.bad .dot{background:var(--red)}.dashboard{display:grid;grid-template-columns:1.25fr .75fr;gap:18px}.panel{background:var(--surface);border:1px solid var(--edge);border-radius:var(--r);padding:22px}.score{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--edge);border:1px solid var(--edge);border-radius:var(--r);overflow:hidden;margin-bottom:18px}.metric{background:var(--surface);padding:20px;min-height:130px}.metric.emphasis{background:var(--raised)}.metric .value{font-size:clamp(2rem,4vw,3.5rem);font-weight:900;font-variant-numeric:tabular-nums}.metric .name{color:var(--muted);margin-top:8px}.metric .delta{font-size:.82rem;margin-top:8px;color:var(--cyan)}.section-head{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:18px}.section-head h2{margin:0;font-size:1.05rem}.chart{height:180px;display:flex;align-items:end;gap:10px;border-bottom:1px solid var(--edge);padding:16px 4px 0}.daybar{height:100%;flex:1;display:flex;flex-direction:column;justify-content:end;align-items:center;gap:7px;min-width:36px}.barcol{width:min(38px,75%);background:var(--cyan);border-radius:5px 5px 0 0;min-height:3px;position:relative}.barcol:after{content:"";position:absolute;inset:0;background:linear-gradient(to top,#0004,transparent)}.daylabel{font:11px monospace;color:var(--muted);white-space:nowrap}.compare{display:grid;grid-template-columns:1fr 1fr;gap:12px}.compare article{padding:18px;background:var(--raised);border-right:3px solid var(--cyan)}.compare strong{font-size:2rem}.facts{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.fact{padding:15px;border-bottom:1px solid var(--edge)}.fact b{display:block;font-size:1.35rem;margin-bottom:4px}.table-wrap{overflow:auto}.wide{grid-column:1/-1}table{width:100%;border-collapse:collapse;min-width:700px}th,td{padding:13px 10px;text-align:right;border-bottom:1px solid var(--edge);font-variant-numeric:tabular-nums}th{font-size:.78rem;color:var(--muted)}.tag{display:inline-block;padding:5px 10px;border:1px solid currentColor;border-radius:99px;font-size:.8rem}.win{color:var(--cyan)}.loss{color:var(--red)}a{color:var(--cyan)}.error{color:var(--red)}.loading{opacity:.6}@media(max-width:900px){.dashboard{grid-template-columns:1fr}.score{grid-template-columns:1fr 1fr}.status-line{grid-template-columns:1fr 1fr}.mast{grid-template-columns:1fr}.live-banner{min-width:0}}@media(max-width:560px){.shell{width:92%}.score,.status-line,.compare{grid-template-columns:1fr}.metric{min-height:auto}.facts{grid-template-columns:1fr}}@media(prefers-reduced-motion:no-preference){.dot{transition:background .2s}.barcol{animation:rise .35s ease-out both}@keyframes rise{from{transform:scaleY(0);transform-origin:bottom}}}
  </style></head><body><main class="shell"><header class="mast"><div class="brand"><div class="mark" aria-hidden="true">M</div><div><div class="kicker">MOSES' STAFF / LIVE INTELLIGENCE</div><h1>غرفة عمليات موسى</h1><div class="muted" id="updated">جلب القياسات الحية…</div></div></div><aside class="live-banner"><strong id="playState">فحص ساحة اللعب…</strong><span class="muted" id="lastActivity">—</span></aside></header><section class="status-line" aria-label="حالة الأنظمة"><div class="status" id="cf"><span class="dot"></span><span>Cloudflare</span></div><div class="status" id="render"><span class="dot"></span><span>Render</span></div><div class="status" id="collector"><span class="dot"></span><span>جامع المباريات</span></div><div class="status" id="trainer"><span class="dot"></span><span>التدريب</span></div></section><section class="score"><article class="metric emphasis"><div class="value" id="todayGames">0</div><div class="name">مباريات اليوم</div><div class="delta" id="todaySplit">—</div></article><article class="metric"><div class="value" id="todayRate">0%</div><div class="name">فوز اليوم</div><div class="delta" id="rateDelta">—</div></article><article class="metric"><div class="value" id="allGames">0</div><div class="name">كل المباريات المسجلة</div><div class="delta" id="record">—</div></article><article class="metric"><div class="value" id="lastTen">0%</div><div class="name">الفوز في آخر 10</div><div class="delta" id="streak">—</div></article></section><section class="dashboard"><article class="panel"><div class="section-head"><h2>نشاط آخر الأيام</h2><span class="muted">عدد المباريات / توقيت بغداد</span></div><div class="chart" id="chart"></div></article><article class="panel"><div class="section-head"><h2>اليوم مقابل الأمس</h2></div><div class="compare"><article><span class="muted">اليوم</span><strong id="compareToday">0%</strong><small id="compareTodayGames">0 مباراة</small></article><article><span class="muted">أمس</span><strong id="compareYesterday">0%</strong><small id="compareYesterdayGames">0 مباراة</small></article></div><div class="facts"><div class="fact"><b id="avgTurns">0</b><span class="muted">متوسط الحركات</span></div><div class="fact"><b id="uptime">0%</b><span class="muted">استقرار الخادم</span></div><div class="fact"><b id="latency">—</b><span class="muted">زمن الاستجابة</span></div><div class="fact"><b id="trainingState">—</b><span class="muted">آخر تدريب</span></div></div></article><article class="panel wide"><div class="section-head"><h2>السجل اليومي الكامل</h2><span class="muted" id="historyCount">—</span></div><div class="table-wrap"><table><thead><tr><th>التاريخ</th><th>المباريات</th><th>الفوز</th><th>الخسارة</th><th>نسبة الفوز</th><th>متوسط الحركات</th></tr></thead><tbody id="history"></tbody></table></div></article><article class="panel wide"><div class="section-head"><h2>آخر المباريات</h2><span class="muted">تُحدّث الصفحة كل دقيقة</span></div><div class="table-wrap"><table><thead><tr><th>النتيجة</th><th>الحركات</th><th>الطول النهائي</th><th>الوقت</th><th>معرّف المباراة</th></tr></thead><tbody id="recent"></tbody></table></div></article></section></main><script>
  const q=s=>document.querySelector(s),fmt=d=>d?new Date(d).toLocaleString('ar-IQ',{timeZone:'Asia/Baghdad'}):'—',ago=d=>{if(!d)return'لا يوجد نشاط مسجل';const m=Math.max(0,Math.round((Date.now()-new Date(d))/60000));return m<1?'قبل أقل من دقيقة':m<60?'قبل '+m+' دقيقة':'قبل '+Math.round(m/60)+' ساعة'},health=(id,ok,label)=>{const e=q(id);e.className='status '+(ok?'ok':'bad');e.querySelector('span:last-child').textContent=label};async function load(){try{const s=await fetch('/api/status',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error(r.status);return r.json()});const m=s.matches,a=s.analytics,t=s.training,u=s.uptime;q('#todayGames').textContent=a.today.total;q('#todaySplit').textContent=a.today.wins+' فوز • '+a.today.losses+' خسارة';q('#todayRate').textContent=a.today.win_rate+'%';q('#rateDelta').textContent=(a.win_rate_change>0?'▲ +':a.win_rate_change<0?'▼ ':'')+a.win_rate_change+' نقطة عن أمس';q('#allGames').textContent=m.total;q('#record').textContent=m.wins+' فوز • '+m.losses+' خسارة';q('#lastTen').textContent=a.last_ten_win_rate+'%';q('#streak').textContent=a.current_streak>0?'سلسلة '+a.current_streak+' انتصارات':a.current_streak<0?'سلسلة '+Math.abs(a.current_streak)+' خسائر':'لا توجد سلسلة';q('#playState').textContent=a.playing_now?'يلعب الآن في '+a.active_games+' مباراة':'بانتظار المباراة التالية';q('#lastActivity').textContent='آخر نشاط '+ago(m.activity?.last_activity_utc||m.last_match_utc);q('#updated').textContent='آخر مباراة '+fmt(m.last_match_utc)+' • آخر فحص '+fmt(u.last_check_utc);q('#compareToday').textContent=a.today.win_rate+'%';q('#compareTodayGames').textContent=a.today.total+' مباراة';q('#compareYesterday').textContent=a.yesterday.win_rate+'%';q('#compareYesterdayGames').textContent=a.yesterday.total+' مباراة';q('#avgTurns').textContent=m.average_turns;q('#uptime').textContent=u.uptime_percentage+'%';q('#latency').textContent=u.last_latency_ms==null?'—':u.last_latency_ms+' ms';const tc=t?.is_training_now?'يعمل الآن':t?.conclusion==='success'?'ناجح':t?.conclusion==='failure'?'فشل':'غير متاح';q('#trainingState').textContent=tc;health('#cf',a.cloudflare_healthy,'Cloudflare يعمل');health('#render',a.render_healthy,'Render '+(a.render_healthy?'يعمل':'متعطل'));health('#collector',a.collector_healthy,'الجمع '+(a.collector_healthy?'سليم':'متوقف'));health('#trainer',t?.status!=='unavailable','التدريب: '+tc);const days=a.history.slice(0,10).reverse(),max=Math.max(1,...days.map(x=>x.total));q('#chart').innerHTML=days.map(x=>'<div class="daybar"><b>'+x.total+'</b><div class="barcol" style="height:'+Math.max(3,x.total/max*120)+'px"></div><span class="daylabel">'+x.date.slice(5)+'</span></div>').join('')||'<span class="muted">بانتظار بيانات أكثر</span>';q('#historyCount').textContent=a.history.length+' أيام مسجلة';q('#history').innerHTML=a.history.map(x=>'<tr><td>'+x.date+'</td><td>'+x.total+'</td><td class="win">'+x.wins+'</td><td class="loss">'+x.losses+'</td><td>'+x.win_rate+'%</td><td>'+(x.average_turns||0)+'</td></tr>').join('')||'<tr><td colspan="6">لا توجد بيانات يومية بعد</td></tr>';q('#recent').innerHTML=m.recent.slice(0,20).map(x=>'<tr><td><span class="tag '+(x.won?'win':'loss')+'">'+(x.won?'فوز':'خسارة')+'</span></td><td>'+x.turns+'</td><td>'+x.final_length+'</td><td>'+fmt(x.ended_at_utc)+'</td><td><code>'+x.game_id+'</code></td></tr>').join('')||'<tr><td colspan="5">لا توجد مباريات مسجلة بعد</td></tr>'}catch(e){q('#updated').innerHTML='<span class="error">تعذر تحميل البيانات: '+e.message+'</span>'}}load();setInterval(load,60000);
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
