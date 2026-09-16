# Moses' Staff Cloudflare keeper

This Worker runs every five minutes, requests the Render API, and stores durable
uptime counters and completed Battlesnake replays in Workers KV. Its public root
URL serves an Arabic match dashboard; `/api/status` returns JSON, `/api/replays`
feeds the learner, and `/check` performs an immediate full refresh.

One-time deployment:

```bash
cd cloudflare/keepalive
npm install
npx wrangler login
npx wrangler kv namespace create STATUS
```

Copy the returned namespace ID into `wrangler.jsonc`, replacing
`REPLACE_WITH_CLOUDFLARE_KV_NAMESPACE_ID`, then run:

```bash
npm run deploy
```

After deployment, open the Worker URL and confirm that its dashboard loads.
The free-plan write budget is protected by writing match and training records
only when they change. The five-minute uptime record is the only unconditional
KV write. This external cron is the primary Render keep-awake mechanism.
