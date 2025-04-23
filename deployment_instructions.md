# PanicSense: Render Deployment Instructions

## Render Dashboard Settings

### Build Command:
```
bash render-deploy.sh
```

### Start Command:
```
node index.js
```

### Environment Variables:
- `NODE_ENV` = `production`
- `DATABASE_URL` = [Your PostgreSQL URL]
- `GROQ_API_KEY` = [Your Groq API Key]

## Critical Files Setup

Three critical files have been created for successful deployment:

1. **render-deploy.sh**: Comprehensive build script that:
   - Cleans previous builds and sets up proper file structure
   - Attempts multiple build strategies for the frontend
   - Handles failures with backup strategies
   - Installs both Node.js and Python dependencies
   - Creates proper directory structures

2. **vite.compat.mjs**: Compatibility config for Vite frontend build on Render

3. **render_setup/render-requirements.txt**: Updated Python requirements compatible with Render environment

## Deployment Process

1. **Push code to GitHub**:
   ```
   git add .
   git commit -m "Add Render deployment configuration"
   git push
   ```

2. **In Render Dashboard**:
   - Set Build Command to: `bash render-deploy.sh`
   - Set Start Command to: `node index.js`
   - Configure environment variables (NODE_ENV, DATABASE_URL, GROQ_API_KEY)

3. **Deploy**:
   - Click "Deploy" button in Render dashboard
   - Wait for the process to complete
   - Check logs for any errors

## Special Features

1. **Fallback Mechanisms**: Backup plans if a step fails
2. **Comprehensive File Structure Setup**: Correct file placement in dist, client/dist, and python folders
3. **Multiple Build Strategies**: Tries several ways to build the frontend
4. **Incremental Deploy**: Backend first, then frontend, to prevent complete failure

## Troubleshooting

If deployment fails:
1. Check Render logs for specific error messages
2. Verify environment variables are correctly set
3. Ensure the database connection is working
4. Check if Groq API key is valid and properly configured