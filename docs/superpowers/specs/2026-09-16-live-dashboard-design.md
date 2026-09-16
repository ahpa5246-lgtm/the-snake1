# Live Snake Dashboard Design

Build a public Cloudflare-hosted dashboard that persists completed Battlesnake
matches, shows daily wins and losses, reports the latest GitHub learning run,
and provides durable replay export for the learner.

The existing five-minute Worker cron will keep Render awake, pull completed
replays, deduplicate them by game ID, store each replay in KV, and update compact
cumulative/daily statistics. The GitHub learner will download the durable Worker
export instead of Render's ephemeral filesystem. Training remains batched every
six hours and retains the benchmark/promotion safety gate.

Endpoints:

- `/` — public Arabic dashboard.
- `/api/status` — JSON containing uptime, match statistics, and latest training run.
- `/api/replays` — bounded completed-replay batch for GitHub Actions.
- `/check` — manual keepalive, replay sync, and training-status refresh.

No credentials are exposed. GitHub workflow state is read from the public
repository API. Match history begins with whatever completed replays are still
available on Render at the first successful synchronization.
