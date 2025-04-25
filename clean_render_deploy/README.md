# PanicSense Render Deployment (ULTRA SIMPLE)

Super simple, direct deployment of PanicSense to Render.com with ZERO complicated scripts.

## Quick Start

1. Copy these files to your repository root:
   - `package.json`
   - `server.js` 
   - `setup.sh`
   - `requirements.txt` (if you need Python)

2. On Render.com:
   - Create a new Web Service 
   - Connect your GitHub repository
   - Set **Build Command**: `npm run build`
   - Set **Start Command**: `npm start`
   - Add required environment variables

## What You Get

- ✅ SIMPLE single-file server that just works
- ✅ No bash scripts or complex build process
- ✅ Auto-fixes for database schema
- ✅ WebSocket support
- ✅ Python compatibility if needed
- ✅ Upload support that works

## Required Environment Variables

- `DATABASE_URL` - Your Postgres database URL 
- `NODE_ENV` - Set to "production"
- `SESSION_SECRET` - Any random string

## How It Works

This setup is designed to be as simple as possible:

1. `npm run build` - Installs dependencies and Python packages
2. `npm start` - Starts the server
3. The server auto-detects:
   - Database schema
   - Available Python scripts
   - Frontend files

There's ZERO complicated logic or scripting - just what's needed for Render.com to work.