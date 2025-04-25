# Render Deployment Fix for PanicSense

This guide provides step-by-step instructions to fix deployment issues on Render.com.

## The Problem

Render.com doesn't install development dependencies by default, which causes errors like `vite: not found` during the build process. Additionally, there are module compatibility issues between CommonJS and ES modules.

## The Solution

We've created several files to fix these issues:

1. `render-build.sh` - Custom build script that installs ALL dependencies
2. `app-render.js` - CommonJS-compatible server for Render deployment

## Deployment Instructions

1. Log in to your Render.com dashboard
2. Create a new Web Service:
   - Connect to your GitHub repository
   - Give it a name (e.g., "PanicSense")
   - Select "Node" as the runtime

3. Configure the build settings:
   - Build Command: `./render-build.sh`
   - Start Command: `node app-render.js`

4. Add the following environment variables:
   - `NODE_ENV`: `production`
   - `PORT`: `10000`
   - `DATABASE_URL`: Your Neon PostgreSQL connection string
   - `SESSION_SECRET`: A random string for session security

5. Click "Create Web Service"

## Troubleshooting Common Issues

### "vite: not found" error
This is caused by Render not installing dev dependencies. Our `render-build.sh` script fixes this by using `npm install --production=false`.

### Module import errors
There can be compatibility issues between CommonJS and ES modules. Our `app-render.js` file is a simplified CommonJS version of the server that avoids these issues.

### Database connection issues
Make sure your `DATABASE_URL` environment variable is correctly set in the Render dashboard. The app will still start without a database connection, but some features won't work.

## Checking Deployment Status

After deployment, visit `/api/health` to check if the server is running correctly:

```
https://your-render-app.onrender.com/api/health
```

This should return a JSON response with the server status.

## Additional Notes

- The production build on Render uses a minimal server configuration without all the backend API routes
- Real-time monitoring features require a valid database connection
- If you need to debug deployment issues, check the logs in the Render dashboard