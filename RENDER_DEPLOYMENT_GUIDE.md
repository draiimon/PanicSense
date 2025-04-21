# PanicSense PH - Render Deployment Guide

This guide provides step-by-step instructions for deploying PanicSense PH to Render.com hosting service. The deployment has been optimized with additional scripts and improvements to ensure proper functionality in the Render environment.

## Required Environment Variables

Ensure you have these environment variables set in your Render dashboard:

| Variable | Value | Description |
|----------|-------|-------------|
| `DATABASE_URL` | `postgres://...` | Your Neon PostgreSQL connection string |
| `NODE_ENV` | `production` | Forces production mode |
| `PORT` | `10000` (or Render default) | Application port |
| `RUNTIME_ENV` | `render` | Enables Render-specific optimizations |
| `DISABLE_SSL_VERIFY` | `true` | Fixes potential SSL issues with API connections |
| `TZ` | `Asia/Manila` | Sets timezone for Philippines |

## Deployment Steps

1. **Create a Web Service in Render Dashboard**
   - Select "Web Service"
   - Connect your GitHub repository
   - Choose the branch with latest fixes (e.g., `render-deployment-fix`)

2. **Configure Build & Start Commands**
   - Build Command: `npm install && npm run build`
   - Start Command: `node pre-render-start.js`

3. **Set Environment Variables**
   - Add all required environment variables listed above
   - Ensure your `DATABASE_URL` is properly formatted with SSL settings

4. **Deploy the Application**
   - Click "Create Web Service"
   - Wait for the build and deployment to complete

## Debugging & Troubleshooting

The enhanced deployment includes several debugging features:

- **Enhanced Logging System**: Logs are now written to files in the `/logs` directory as well as to the console
- **Automatic Database Fixes**: Detects and resolves common database issues
- **Python Integration Improvements**: Better detection of Python paths and scripts
- **Scheduled Jobs**: Periodic tasks run to keep services active and logs visible
- **Directory Checks**: Validates critical directories and file permissions

If you encounter issues, check the logs in the Render dashboard. The enhanced logging should provide more detailed information about any problems.

## Accessing Log Files

Logs are saved in the `/logs` directory. You can view them through the Render shell if needed:

```bash
# In Render shell
cd logs
ls -la
cat render-startup-*.log
```

## Monitoring Services

Important services that should be active:
- News feed service (fetches disaster news every 10 minutes)
- Python sentiment analysis service
- Database connection
- Static file serving 

Each service now logs its status regularly to make issues more visible.

## Issues Fixed in Render Deployment

1. **Python Path Detection**: Better handling of Python binary and script paths
2. **Enhanced Error Logging**: More detailed error reporting for all services
3. **Fixed Database Connection**: Better SSL handling for PostgreSQL
4. **Directory Structure**: Automatic creation of required directories
5. **Static File Handling**: Improved handling of static assets

## Ensuring CSV Processing Works

CSV processing requires proper Python setup in the Render environment. The following has been configured:

1. Python is automatically located in the Render environment
2. Temporary directories are properly created with correct permissions
3. Process timeouts have been extended for larger files
4. Error handling has been improved for better visibility

## Setting Up Real News and Social Media

The news feed system has been configured to:

1. Fetch news from reliable Philippine sources periodically
2. Log all fetched items for better visibility
3. Automatically retry on temporary failures
4. Run on a scheduled basis to ensure fresh data

## Troubleshooting Common Issues

If you encounter any of these issues:

1. **Blank screen or app not loading**: Check if static files were properly copied to `server/public`
2. **Database errors**: Ensure `DATABASE_URL` is correct and includes SSL parameters
3. **Missing Python functionality**: Check logs for Python path detection issues
4. **No news data**: Look for news feed service logs to see if feeds are being fetched

For any other issues, check the enhanced logs which should provide more detailed information about the problem.