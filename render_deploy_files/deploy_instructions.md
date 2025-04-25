# PanicSense Render Deployment Instructions

## How to Deploy to Render using these files

1. **Push these files to GitHub:**

   ```bash
   # Create a new branch (if needed)
   git checkout -b render-deploy

   # Copy these files to your repository root
   cp -r render_deploy_files/* /path/to/your/repo/

   # Add, commit, and push
   cd /path/to/your/repo
   git add .
   git commit -m "Add clean Render deployment files"
   git push origin render-deploy
   ```

2. **On Render.com:**

   - Create a new Web Service
   - Connect your GitHub repository 
   - Select the branch with these files
   - Set the following:
     - Build Command: `./render-build.sh`
     - Start Command: `node start-render.cjs`

3. **Set Required Environment Variables:**

   - `DATABASE_URL` - Your PostgreSQL database URL
   - `NODE_ENV` - Set to `production`
   - `SESSION_SECRET` - Any random secure string
   - `DEBUG` - Set to `true` for detailed logs (optional)

## Files and Their Purpose

- `production-server.cjs` - Combined Node.js + Python server
- `start-render.cjs` - Simple startup script
- `render-build.sh` - Build script for Render
- `python/daemon.py` - Python daemon that runs without arguments
- `requirements.txt` - Python dependencies
- `package.json` - Minimal Node.js dependencies
- `RENDER-DEPLOY-FIX.md` - Detailed troubleshooting guide
- `README.md` - General project information

## Features

- Real-time analysis
- News feeds and disaster alerts
- WebSocket support
- Automatic Python restart
- File upload functionality