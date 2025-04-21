# PanicSense Render.com Deployment Instructions

## Problema
Ang aplikasyon ay nag-error sa Render.com dahil sa `vite not found` issue at mismatch ng ES Modules at CommonJS.

## Solusyon
1. **Para mag-deploy sa Render.com:**

   - I-upload ang mga files na ito sa iyong repository:
     - `Dockerfile.new` → i-rename bilang `Dockerfile` 
     - `start.sh.new` → i-rename bilang `start.sh`
     - `render.yaml`

2. **Sa Render.com Dashboard:**

   - **Environment:** Docker
   - **Plan:** Free
   - **Region:** Singapore
   - **Repository:** https://github.com/draiimon/PanicSense
   - **Branch:** main
   - **Dockerfile Path:** ./Dockerfile  
   - **Build Context Directory:** .
   - **Docker Command:** (leave blank - gagamitin ang CMD sa Dockerfile)

3. **Environment Variables:**

   - `NODE_ENV` = production
   - `PORT` = 5000
   - `TZ` = Asia/Manila 
   - `DATABASE_URL` = (i-secure ito bilang Secret)
   - `DB_CONNECTION_RETRY_ATTEMPTS` = 5
   - `DB_CONNECTION_RETRY_DELAY_MS` = 3000

4. **Health Check Path:** 
   - `/api/health`
   
5. **Auto-Deploy:** 
   - Enabled

## Paano Gagana?
Ang bagong setup ay:

1. Gumagamit ng multi-stage Docker build
2. Una, binubuo ang React app at node modules sa build stage
3. Pagkatapos, kino-copy lang ang mga pre-built files sa final image
4. Direct na tumatakbo ang `server.js` (CommonJS) sa halip na `index.js` (ES Modules)
5. Inaalis ang problema sa `vite not found` sa pamamagitan ng pag-build sa client sa build stage

Pagkatapos i-push ang mga changes, pindot ang "Manual Deploy" at aabangan ang successful deployment.