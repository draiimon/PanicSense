# Render Auto-Deployment Setup Instructions

This document provides instructions for setting up automatic deployment to Render.com from GitHub.

## GitHub Actions Workflow Setup

1. In your GitHub repository, go to the "Actions" tab
2. Click "New workflow"
3. Choose "set up a workflow yourself"
4. Name the file `render-deploy.yml`
5. Copy and paste the following content:

```yaml
name: Deploy to Render

on:
  push:
    branches:
      - main  # Adjust this to your main branch name if different

jobs:
  deploy:
    name: Deploy to Render
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Deploy to Render
        uses: JorgeLNJunior/render-deploy@v1.4.3
        with:
          service_id: ${{ secrets.RENDER_SERVICE_ID }}  # Your Render service ID
          api_key: ${{ secrets.RENDER_API_KEY }}  # Your Render API key
          clear_cache: true
          wait_deploy: true
          github_token: ${{ secrets.GITHUB_TOKEN }}
```

6. Click "Commit changes" to add this file to your repository

## Setting Up GitHub Secrets

1. Go to your GitHub repository
2. Click on "Settings" > "Secrets and variables" > "Actions"
3. Click "New repository secret"
4. Add the following secrets:
   - Name: `RENDER_SERVICE_ID`
     - Value: Your Render service ID (found in your service URL or settings)
   - Name: `RENDER_API_KEY`
     - Value: Your Render API key (generated in Render dashboard)

## Render Setup

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the following settings:
   - **Build Command**: `./render-build.sh`
   - **Start Command**: `./render-start.sh`
   - **Environment Variables**:
     - `NODE_ENV` = `production`
     - `DATABASE_URL` = (Connect to your Render PostgreSQL database)

## Getting Your Render API Key and Service ID

1. **API Key**:
   - Log in to your Render dashboard
   - Go to "Account Settings" > "API Keys"
   - Create a new API key and save it securely

2. **Service ID**:
   - Open your service in Render dashboard
   - The service ID is in the URL:
   - Example: `https://dashboard.render.com/web/srv-abc123` (srv-abc123 is your service ID)

Once these are set up, every push to your main branch will automatically trigger a deployment to Render.

## Files Included in This Setup

This repository includes several files for deployment:

- `Dockerfile` - Docker container configuration
- `docker-compose.yml` - Local development setup with Docker
- `.dockerignore` - Excludes unnecessary files from Docker builds
- `render.yaml` - Render Blueprint configuration
- `render-build.sh` - Build script for Render deployment
- `render-start.sh` - Start script for Render deployment

These files make the deployment process seamless whether you're using GitHub Actions or manual deployment.