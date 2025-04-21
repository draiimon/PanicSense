# PanicSense PH - Render Setup Instructions

## Deployment Configuration

Since we cannot modify the package.json file directly, follow these specific instructions for deploying to Render:

### Build Command
Use this exact build command in your Render Web Service configuration:

```
npm install && npm run build && node render-fix-python-paths.js
```

This will:
1. Install dependencies
2. Build the application
3. Run the Python path fixer script

### Start Command
Use this exact start command in your Render Web Service configuration:

```
node pre-render-start.js
```

This will:
1. Set up the environment
2. Fix paths and permissions
3. Start the application with enhanced logging

## Environment Variables

Add these environment variables in your Render dashboard:

| Variable | Value | Description |
|----------|-------|-------------|
| `DATABASE_URL` | `postgres://...` | Your Neon PostgreSQL connection string |
| `NODE_ENV` | `production` | Forces production mode |
| `PORT` | `10000` (or Render default) | Application port |
| `RUNTIME_ENV` | `render` | Enables Render-specific optimizations |
| `DISABLE_SSL_VERIFY` | `true` | Fixes potential SSL issues with API connections |
| `TZ` | `Asia/Manila` | Sets timezone for Philippines |

## Step-by-Step Deployment Process

1. Log in to your Render dashboard
2. Click "New" and select "Web Service"
3. Connect to your GitHub repository
4. Select the branch with the latest fixes
5. Configure the service with:
   - Name: PanicSense
   - Region: Singapore (or closest to Philippines)
   - Branch: your branch with fixes
   - Root Directory: leave blank
   - Runtime: Node
   - Build Command: `npm install && npm run build && node render-fix-python-paths.js`
   - Start Command: `node pre-render-start.js`
6. Add all the environment variables listed above
7. Select the appropriate plan
8. Click "Create Web Service"

## Monitoring the Deployment

After deploying, you'll be able to see more detailed logs in the Render dashboard's "Logs" tab. The enhanced logging will show:

- Real-time news fetch activity
- Python service activity
- Database connection status
- File and directory operations

## Troubleshooting Steps

If you encounter deployment issues:

1. Check the logs in the Render dashboard
2. Verify all environment variables are set correctly
3. Check that the database connection is working
4. Ensure Python is properly detected (logs will show this)

For any further issues, the detailed logging in `/logs` directory will help identify the problem.