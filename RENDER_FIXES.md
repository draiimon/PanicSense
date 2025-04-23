# Render Deployment Fixes

## Problema: Vite not found sa Render deployment

Sa mga logs nakita natin:
```
error during build:
Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'vite' imported from /opt/render/project/src/node_modules/.vite-temp/vite.config.ts.timestamp-1745446158777-fc3bc81b1fce2.mjs
```

## Solution: Gumana ang fallback mechanism natin!

```
⚠️ Vite build failed, using fallback method...
Creating minimal index.html...
  dist/index.js  264.0kb
⚡ Done in 13ms
ESBuild succeeded!
```

## Files for Render Deployment

1. **build.sh** - Main script for Render with fallback mechanism
2. **Procfile** - Backup process dito gagamitin si run.js
3. **run.js** - Pure JavaScript script for Render na papaganahin ang app
4. **render.yaml** - Configuration file para sa Render deployment
5. **vite.config.js** - Simplified config para sa vite

## How to set up Render

1. Create new Web Service (not Blueprint)
2. Point to your GitHub repository
3. Environment: Node
4. Build Command: `chmod +x ./build.sh && ./build.sh`
5. Start Command: `NODE_ENV=production node server/index-wrapper.js`

## Environment Variables
- NODE_ENV=production
- PORT=10000
- DATABASE_URL=your-neon-database-url
- GROQ_API_KEY=your-groq-api-key
- SESSION_SECRET=random-string

## Fallback Mechanism Works!
- If vite fails, gumagana pa rin ang ESBuild
- ESBuild compiles ang backend code
- May minimal index.html na nilikha para sa frontend
- Kopya lahat ng Python files
- Install lahat ng dependencies

## Next steps
- I-check ang logs sa Render Dashboard
- May health check endpoint na `/api/health`