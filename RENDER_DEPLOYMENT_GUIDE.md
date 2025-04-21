# Professional Deployment Guide for PanicSense PH on Render.com

This guide provides detailed, step-by-step instructions for deploying the PanicSense PH application to Render.com without using Blueprints (no credit card required).

## Pre-Deployment Checklist ✅

Verify these requirements before deployment:

1. **Github Repository**: Complete code pushed to a GitHub repository
2. **Docker Configuration**: Optimized Dockerfile is present in repository
3. **Database Structure**: Drizzle ORM schemas prepared in `/shared/schema.ts`
4. **Environment Variables**: Prepare required environment variables (listed below)

## Deployment Process

### Step 1: Setup Database on Render

1. Log in to your [Render Dashboard](https://dashboard.render.com)
2. Select **New** and choose **PostgreSQL**
3. Configure the database:
   - **Name**: `panicsense-database` (or your preferred name)
   - **Database**: `panicsense` (default database name)
   - **User**: Auto-generated (record for later use)
   - **Region**: Southeast Asia (closest to Philippines)
   - **PostgreSQL Version**: 15 (recommended for best compatibility)
   - **Instance Type**: Free tier is sufficient for testing
4. Click **Create Database**
5. On the database info page, note down:
   - **Internal Database URL** (for use in web service)
   - **External Database URL** (for administration)
   - **Username** and **Password**

### Step 2: Deploy Web Service on Render

1. From your Render Dashboard, select **New** and choose **Web Service**
2. Connect your GitHub repository
3. Configure the Web Service:
   - **Name**: `panicsense-ph` (or your preferred name)
   - **Region**: Southeast Asia (closest to Philippines)
   - **Branch**: `main` (or your primary branch)
   - **Runtime**: `Docker`
   - **Instance Type**: Free (for testing) or Standard (for production)
4. Add environment variables:
   - `DATABASE_URL`: *Copy the Internal Database URL from Step 1*
   - `NODE_ENV`: `production`
   - `PORT`: `5000`
   - `PYTHON_PATH`: `python3`
   - `TZ`: `Asia/Manila`
   - `SESSION_SECRET`: Generate a strong random string (you can use: `openssl rand -base64 32`)
5. Optional: If you're experiencing database connection delays during startup, add:
   - `DB_CONNECTION_RETRY_ATTEMPTS`: `5`
   - `DB_CONNECTION_RETRY_DELAY_MS`: `3000`
6. Click **Create Web Service**

### Step 3: Monitor Deployment

1. Render will build your Docker image (this may take 5-10 minutes)
2. Monitor the build logs for any errors
3. Once deployed, Render provides a URL (e.g., `https://panicsense-ph.onrender.com`)
4. Verify the application is working by accessing the provided URL

### Step 4: Post-Deployment Tasks

1. **Test the Application**:
   - Verify all pages load correctly
   - Test data uploads and analysis functionality
   - Ensure real-time monitoring works

2. **Configure Custom Domain** (optional):
   - In your Web Service settings, go to "Custom Domain"
   - Add your domain and follow Render's instructions for DNS setup

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgres://user:password@host:port/database` |
| `NODE_ENV` | Environment mode | `production` |
| `PORT` | Application port | `5000` |
| `PYTHON_PATH` | Path to Python executable | `python3` |
| `TZ` | Timezone for logs and data | `Asia/Manila` |
| `SESSION_SECRET` | Secret for session security | `a-very-long-secure-random-string` |

## Troubleshooting

### Database Connection Issues

If you encounter database connection errors:

1. Verify the `DATABASE_URL` is correct in your environment variables
2. Ensure your database is running (check status in Render Dashboard)
3. Check if your IP is whitelisted in Render's database settings

### Build Failures

If your Docker build fails:

1. Check build logs for specific errors
2. Verify Docker image configurations
3. Ensure all required files are in your repository

### Application Crashes

If the application crashes after deployment:

1. Check logs in the Render Dashboard
2. Verify all environment variables are properly set
3. Ensure the database migration has completed successfully

## Maintenance

For ongoing maintenance:

1. **Scaling**: Adjust instance type as needed in Render Dashboard
2. **Updates**: Push updates to your GitHub repository, and Render will automatically rebuild
3. **Monitoring**: Use Render's built-in monitoring for resource usage

## Reminder

* The free tier of Render has resource limitations and will automatically spin down after periods of inactivity
* Your application will spin up again when accessed, but this may take 30-60 seconds
* For production use, consider using the Standard tier to avoid spin-down

---

For more information, visit the [Render Documentation](https://docs.render.com/web-services)