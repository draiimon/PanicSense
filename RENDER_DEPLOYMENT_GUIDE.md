# PanicSense Deployment Guide for Render.com

This guide provides step-by-step instructions for deploying PanicSense to Render.com using the free tier.

## Prerequisites

1. A Render.com account
2. A PostgreSQL database (either on Render or elsewhere like Neon.tech)
3. Git repository for your PanicSense project

## Deployment Steps

### 1. Set Up Your Database

**If using Render's PostgreSQL:**

1. Log in to your Render dashboard
2. Navigate to "Databases" on the left sidebar
3. Click "New Database"
4. Select PostgreSQL
5. Fill in:
   - Name: `panicsense-db`
   - Database: `panicsense`
   - User: Leave as default
   - Region: Choose a region close to your users (e.g., Singapore for PH users)
   - Plan: Free tier
6. Click "Create Database"
7. Once created, note your Internal Database URL

**If using Neon.tech or other PostgreSQL provider:**

1. Create a PostgreSQL database with your provider
2. Note your database connection string

### 2. Deploy to Render

#### Option 1: Direct from GitHub

1. Log in to your Render dashboard
2. Navigate to "Web Services" on the left sidebar
3. Click "New Web Service"
4. Connect your GitHub repository
5. Configure your web service:
   - Name: `panicsense-ph`
   - Environment: Docker
   - Region: Singapore (or closest to your users)
   - Branch: `main` (or your default branch)
   - Plan: Free
6. Under "Advanced" settings, add environment variables:
   - `DATABASE_URL`: Your database connection string
   - (All other variables are already defined in render.yaml)
7. Click "Create Web Service"

#### Option 2: Using render.yaml (Blueprint)

1. Log in to your Render dashboard
2. Navigate to "Blueprints" on the left sidebar
3. Click "New Blueprint Instance"
4. Connect your GitHub repository
5. Render will automatically detect the render.yaml file
6. Add your required environment secrets:
   - `DATABASE_URL`: Your database connection string
7. Click "Apply"

### 3. Setup SSL and Custom Domain (Optional)

If you have a custom domain for your PanicSense application:

1. From your web service page, click on "Settings"
2. Scroll to "Custom Domains"
3. Click "Add Custom Domain"
4. Enter your domain name (e.g., panicsense.ph)
5. Follow the DNS configuration instructions
6. Render will automatically provision a free SSL certificate

### 4. Verify Deployment

1. Once deployment is complete, click on your service URL
2. Verify that the application is working correctly
3. Check the logs for any errors or warnings

## Troubleshooting

### Database Connection Issues

If your application cannot connect to the database:

1. Verify your DATABASE_URL is correct
2. Make sure SSL is enabled (should be handled by start.sh)
3. Check that your database is accessible from Render

### Build Failures

If your Docker build fails:

1. Check Render logs for specific error messages
2. Verify that the Dockerfile is correctly set up
3. Make sure your repository doesn't exceed size limits

### Application Errors

If the application deploys but doesn't work correctly:

1. Check Render logs for runtime errors
2. Verify all required environment variables are set
3. Make sure your database schema is correctly set up

## Maintenance

### Updating Your Application

1. Push changes to your GitHub repository
2. Render will automatically deploy updates (if auto-deploy is enabled)

### Database Backups

If using Render PostgreSQL:

1. Backups are automatically created daily
2. You can create manual backups from the database dashboard

## Support

If you encounter issues with your Render deployment, you can:

1. Check Render documentation at https://render.com/docs
2. Contact Render support through your dashboard
3. Review application logs in the Render dashboard

For application-specific issues, please refer to the PanicSense documentation or contact the development team.