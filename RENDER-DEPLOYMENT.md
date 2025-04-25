# PanicSense Render Deployment Guide

This guide will walk you through deploying PanicSense to Render.com.

## Prerequisites

1. A Render.com account
2. A PostgreSQL database (you can use Render's PostgreSQL service or an external provider like Neon)

## Setup Steps

### 1. Create a Web Service in Render

1. Go to the Render dashboard and click **New+** → **Web Service**
2. Connect your repository or provide the GitHub URL
3. Fill in the following details:
   - **Name**: `panicsense` (or your preferred name)
   - **Region**: Choose a region close to your users
   - **Branch**: `main` (or your deployment branch)
   - **Runtime**: `Node`
   - **Build Command**: `./render-build-bash.sh`
   - **Start Command**: `node start-render.js`

### 2. Configure Environment Variables

Add these environment variables in the Render dashboard:

```
NODE_ENV=production
PORT=10000
DATABASE_URL=your_postgres_connection_string
SESSION_SECRET=your_secure_random_string
```

Replace `your_postgres_connection_string` with your actual PostgreSQL connection string.

If you're using Neon database, you can also set:

```
NEON_DATABASE_URL=your_neon_connection_string
```

### 3. Deploy

Click the **Create Web Service** button to start the deployment process.

## Troubleshooting

If you run into deployment issues:

1. Check the Render logs for specific error messages
2. Ensure your database connection string is correct
3. Verify that all environment variables are set
4. If you're using Neon database, make sure the WebSocket connection is properly configured

## Understanding the Deployment

The deployment uses special scripts to ensure compatibility:

1. `render-build-bash.sh`: Builds the application with necessary dependencies
2. `start-render.js`: Starts the server with proper configuration
3. `start-render.cjs`: Contains the actual server startup logic in CommonJS format

If you need to modify the deployment process, these are the key files to update.