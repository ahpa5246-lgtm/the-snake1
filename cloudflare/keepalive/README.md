# Moses' Staff Cloudflare keeper

This Worker runs every five minutes, requests the Render API, and stores durable
uptime counters in Workers KV. Its public root URL returns the current status as
JSON; `/check` performs an immediate check and returns the updated counters.

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

After deployment, open the Worker URL once and confirm that `total_checks`
increases every five minutes. This external cron is the primary Render
keep-awake mechanism; GitHub's scheduled keep-alive remains a fallback.
