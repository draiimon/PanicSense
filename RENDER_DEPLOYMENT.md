# Render Deployment Guide for PanicSense PH

This guide will walk you through deploying the PanicSense PH application to Render.com.

## Prerequisites

1. A Render.com account
2. Your project code pushed to a Git repository (GitHub, GitLab, or Bitbucket)
3. API keys for any third-party services used by the application (if applicable)

## Deployment Steps

### 1. Create a New Web Service on Render

1. Log in to your Render account
2. Click on "New" and select "Blueprint" from the dropdown menu
3. Connect your Git repository containing the application code
4. Render will automatically detect the `render.yaml` configuration file

### 2. Configure Environment Variables

While the `render.yaml` file defines most environment variables, you'll need to set up any API keys or secrets manually:

1. Go to your web service dashboard on Render
2. Click on "Environment" in the left sidebar
3. Add any required API keys (such as OPENAI_API_KEY if you're using OpenAI services)
4. Click "Save Changes"

### 3. Database Configuration

The `render.yaml` file already specifies a PostgreSQL database. Render will:

1. Create a new PostgreSQL database for your application
2. Automatically inject the database connection string into your application
3. Run migrations using the `render-setup.sh` script during the startup process

### 4. Monitor the Deployment

1. Go to the "Logs" tab to watch the deployment process
2. Check for any errors during the build or startup process
3. Once complete, your application will be available at your Render URL

### 5. Custom Domain Setup (Optional)

To use a custom domain:

1. Go to the "Settings" tab of your web service
2. Under "Custom Domain", click "Add Custom Domain"
3. Follow the instructions to configure your DNS settings

## Troubleshooting

If you encounter issues during deployment:

1. Check the logs for error messages
2. Verify your environment variables are correctly set
3. Make sure your database connection string is valid
4. Confirm that any required API keys are provided

## Post-Deployment

After successful deployment:

1. Test your application thoroughly
2. Set up monitoring and alerts
3. Configure automatic backups for your database

## Resource Management

Render's free tier has limitations. To optimize your application:

1. Monitor your resource usage in the Render dashboard
2. Upgrade to paid plans if necessary for better performance
3. Configure auto-scaling if you expect high traffic

## Security Considerations

1. Do not store API keys or secrets in your codebase
2. Use environment variables for sensitive information
3. Keep your dependencies updated
4. Enable automatic HTTPS for your service (Render provides this by default)