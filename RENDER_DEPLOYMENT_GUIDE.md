# PanicSense Render Deployment Guide

This guide provides instructions for deploying PanicSense to Render.com.

## Pre-Deployment Checklist

Before deploying, verify:

1. **Database Connection**: Neon.tech or compatible PostgreSQL database is ready
2. **Environment Variables**: Your `.env` file has all required variables
3. **Branch Ready**: Your code is on the `render-deployment-fix` branch

## Deployment Steps

### 1. Creating a Web Service

1. Log into your [Render Dashboard](https://dashboard.render.com/)
2. Click **New** and select **Web Service**
3. Connect your GitHub repository (or use **Deploy from Existing Repository** option)
4. Configure the following settings:
   - **Name**: `PanicSense` or your preferred name
   - **Root Directory**: _Leave empty_
   - **Environment**: `Node`
   - **Region**: Select the region closest to the Philippines
   - **Branch**: `render-deployment-fix`
   - **Build Command**: `npm ci && npm run build`
   - **Start Command**: `./start.sh`
   - **Plan**: Free tier (or paid tier for better performance)

### 2. Environment Variables

Add these environment variables in the Render Dashboard:

| Key | Value | Description |
|-----|-------|-------------|
| `DATABASE_URL` | `postgres://...` | Your Neon or PostgreSQL database URL |
| `NODE_ENV` | `production` | Run in production mode |
| `PORT` | `5000` or `10000` | Port for the application (typically 10000 for Render) |
| `TZ` | `Asia/Manila` | Philippine timezone |
| `RUNTIME_ENV` | `render` | Tells app it's running on Render |
| `DB_SSL_REQUIRED` | `true` | Ensures SSL for database connection |
| `HOST` | `0.0.0.0` | Listen on all interfaces |

### 3. Database Setup

If you're using a Neon.tech or PostgreSQL database:

1. Make sure your database connection string is added to the environment variables
2. Tables will be created automatically on first run through the render-setup.js script
3. If tables don't create automatically, you may need to manually push the schema from your local environment first

## Troubleshooting

### Common Issues

#### Build Fails due to Node Version

**Issue**: Build fails with node version errors
**Solution**: Add `.node-version` file with content `20.x` to the root of your project

#### Database Connection Fails

**Issue**: Database connection errors in logs
**Solutions**:
- Verify your DATABASE_URL is correct and includes SSL parameters
- Check if your database allows connections from Render's IP addresses
- Test connection manually using `psql` from your local machine

#### ES Module / CommonJS Errors

**Issue**: Error messages about `require` not being defined in ES modules
**Solutions**:
- Ensure server.js is using import statements instead of require
- Check that package.json has `"type": "module"`
- For mixed module types, use .cjs extension for CommonJS files
- Update any dynamic imports to use async/await pattern:
  ```javascript
  // Instead of this:
  const { something } = require('./file');
  
  // Use this:
  const module = await import('./file.js');
  const { something } = module;
  ```

#### Static Files Not Loading

**Issue**: Frontend shows but without CSS/JS or shows a blank page
**Solutions**:
- Check logs to see if render-setup.js ran correctly
- Verify client/dist or dist/public files were copied to server/public
- Ensure server.js has the correct path to static files
- Check browser console for 404 errors on specific files

## Deployment Verification

After deployment:

1. Check the **Logs** tab in your Render dashboard
2. Verify your application is running (look for "🚀 Server running on port...")
3. Click the generated domain name to open your application
4. Test core functionality to ensure everything works

## Updating Your Deployment

To update your deployed application:

1. Push changes to the `render-deployment-fix` branch
2. Render will automatically detect the changes and deploy

## Support

If you encounter issues not covered in this guide, please:

1. Check Render's logs for specific error messages
2. Review the Render documentation at https://render.com/docs
3. Check for common Node.js deployment issues in the troubleshooting section above