# PanicSense Render Deployment

## Clean Deployment Files for Render.com

These are the essential files needed to deploy PanicSense on Render.com's free tier without requiring a credit card or blueprints.

## Files and Their Purpose

- `production-server.cjs` - Combined Node.js + Python server
- `start-render.cjs` - Startup script for Render
- `render-build.sh` - Build script for Render
- `python/daemon.py` - Auto-running Python script (no arguments needed)
- `package.json` - Minimal dependencies for deployment
- `RENDER-DEPLOY-FIX.md` - Detailed deployment instructions

## How to Deploy

1. On Render.com, create a new Web Service
2. Connect your GitHub repository or upload these files
3. Configure using these settings:
   - Build Command: `./render-build.sh`
   - Start Command: `node start-render.cjs`
4. Add these environment variables:
   - `DATABASE_URL` - Your Neon PostgreSQL database URL
   - `NODE_ENV` - Set to `production`
   - `SESSION_SECRET` - Any secure random string

## Features Supported

- ✅ Real-time analysis with Python integration
- ✅ File uploads and processing
- ✅ News feeds and disaster alerts
- ✅ Web interface with API endpoints
- ✅ Automatic Python restart if it crashes
- ✅ WebSocket support for real-time updates