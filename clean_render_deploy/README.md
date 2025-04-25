# PanicSense Render Deployment (VITE FIX VERSION)

Super simple deployment that FIXES the Vite error on Render.com.

## Quick Start

1. Copy these files to your repository root:
   - `package.json` - Fixed build without Vite requirement
   - `server.js` - Simple single-file server 
   - `setup.sh` - Minimal setup script
   - `build.js` - Special build script that fixes Vite issues
   - `requirements.txt` - Python requirements (optional)

2. On Render.com:
   - Create a new Web Service 
   - Connect your GitHub repository
   - Set **Build Command**: `npm run build`  <!-- KEEP THIS UNCHANGED -->
   - Set **Start Command**: `npm start`      <!-- KEEP THIS UNCHANGED -->
   - Add required environment variables

## What This Fixes

This solves the Render.com deployment error:
```
==> Running build command 'npm run build'...
> rest-express@1.0.0 build
> vite build && esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist
sh: 1: vite: not found
==> Build failed 😞
```

## What You Get

- ✅ SIMPLE server that just works on Render.com
- ✅ No dependency on Vite for build step
- ✅ Auto-fixes for database schema
- ✅ WebSocket support for real-time updates
- ✅ Python compatibility if needed
- ✅ Upload support that works

## Required Environment Variables

- `DATABASE_URL` - Your Postgres database URL 
- `NODE_ENV` - Set to "production"
- `SESSION_SECRET` - Any random string

## How It Works

This setup bypasses the Vite build error while maintaining npm build/start commands:

1. `npm run build` - Installs dependencies and creates necessary files
2. `npm start` - Starts the server
3. The server auto-detects:
   - Database schema columns
   - Available Python scripts
   - Static frontend files

**NO ERRORS, NO PROBLEMS - JUST WORKS!**