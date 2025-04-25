# Complete Render Deployment for PanicSense

This guide provides step-by-step instructions for deploying the FULL application on Render.com, including Python services, real-time analysis, and file uploads.

## Requirements

The full PanicSense application requires:

1. Node.js server with development mode
2. Python backend for NLP processing
3. Database connection for storing results
4. File upload functionality

## Complete Deployment Files

We've created special files to handle a full deployment:

1. `render-build-complete.sh` - Build script that installs both Node and Python dependencies
2. `app-render-complete.cjs` - Full server with Python integration in development mode

## Deployment Instructions

1. Log in to your Render.com dashboard
2. Create a new Web Service:
   - Connect to your GitHub repository
   - Give it a name (e.g., "PanicSense")
   - Select "Node" as the runtime

3. Configure the build settings:
   - Build Command: `./render-build-complete.sh`
   - Start Command: `node app-render-complete.cjs`

4. Add the following environment variables:
   - `NODE_ENV`: `development`
   - `PORT`: `10000`
   - `DATABASE_URL`: Your Neon PostgreSQL connection string
   - `SESSION_SECRET`: A random string for session security
   - `PYTHON_PATH`: `python` (or path to Python 3 on Render)

5. Click "Create Web Service"

## What This Deployment Includes

Unlike the minimal deployment, this complete setup includes:

- ✅ Python backend services
- ✅ Real-time news monitoring
- ✅ File upload functionality
- ✅ Disaster event tracking
- ✅ Sentiment analysis capabilities
- ✅ Geographic visualization
- ✅ Development mode for better debugging

## Verifying It Works

After deployment, check the following endpoints:

1. `/api/health` - Should show Python is active
2. `/api/disaster-events` - Should show disaster events from the database
3. Upload functionality should work on the web interface

## Troubleshooting

### Python not starting
If you see "Python process exited with code 1" in the logs with an error about missing arguments, it means the Python daemon script isn't running properly. Check the Render logs for Python-related errors.

We've included a special daemon.py script that runs without requiring command-line arguments, which should solve the most common issue. Make sure `PYTHON_PATH` is correctly set in your environment variables.

### Database connection issues
Verify your DATABASE_URL is correctly set in the Render environment variables. The app uses Neon PostgreSQL.

### File upload issues
Ensure the permissions in Render allow file system access. Check if the uploads directory is being created.

## Advanced Configuration

For more customization, you can add these environment variables:

- `NEWS_UPDATE_INTERVAL`: Time in minutes between news updates (default: 15)
- `DEBUG`: Set to "true" for verbose logging
- `MAX_UPLOAD_SIZE_MB`: Maximum upload file size in MB (default: 50)

## Monitoring

Render provides logs and metrics for your application. Check the Render dashboard for:

- CPU and memory usage
- Request logs
- Build and deployment history