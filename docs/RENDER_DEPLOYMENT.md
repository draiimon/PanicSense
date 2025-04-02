# Render Deployment Guide

This guide will walk you through deploying PanicSense to Render and troubleshooting common issues.

## Deployment Steps

1. **Push changes to your GitHub repository**

2. **Connect to Render**
   - If you're using the Render Blueprint (render.yaml), Render will automatically deploy your app
   - Otherwise, create a new Web Service pointing to your GitHub repository

3. **Configure environment variables**
   - Database connection will be automatically configured if using Render PostgreSQL
   - Add your Groq API keys manually:
     - `VALIDATION_API_KEY` - Single API key used for sentiment validation
     - `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, etc. - Multiple API keys for rotation

## Latest Changes (April 2, 2025)

We've made several improvements to ensure smooth deployment on Render:

1. **Enhanced Database Migration**
   - Added comprehensive schema creation script (`migrations/complete_schema.sql`)
   - Updated migration runner to handle various database connection scenarios
   - Added quick-fix script for emergency fixes (`quick-fix.sh`)

2. **Dockerfile Optimization**
   - Added migration script execution before application start
   - Improved error handling for database connections

## Troubleshooting Common Issues

### Database Schema Problems

If you see errors like `relation "training_examples" does not exist` or `column "ai_trust_message" of relation "sentiment_posts" does not exist`:

1. Go to your Render dashboard
2. Open the Web Service (disaster-monitoring-app)
3. Go to the "Shell" tab
4. Run: `bash quick-fix.sh`
5. Restart the service

For more detailed database instructions, see `database-setup.md`.

### API Key Issues

If you see errors related to Groq API or sentiment analysis failing:

1. Ensure all required environment variables are set in the Render dashboard
2. Check that your API keys have sufficient credits
3. You can test a specific API key by using the Shell tab and running:
   ```
   curl -H "Authorization: Bearer $GROQ_API_KEY_1" https://api.groq.com/v1/models
   ```

## Development vs Production

There are a few key differences between local development and Render deployment:

1. **Environment**
   - Local: `NODE_ENV=development`, uses Vite for development server
   - Render: `NODE_ENV=production`, serves static built files

2. **Database**
   - Local: Usually connects without SSL
   - Render: Uses SSL for database connections

3. **Port Binding**
   - The app internally uses port 5000
   - Render automatically maps this to the appropriate external port

## Monitoring

Render provides built-in logs to help diagnose issues:

1. Go to your Web Service in the Render dashboard
2. Click on "Logs" to view application output
3. Set the log level to "Debug" for more detailed information

For more information, refer to the [Render Documentation](https://render.com/docs).