# Disaster Monitoring Platform

An advanced AI-powered disaster monitoring and community resilience platform for the Philippines.

## Deployment Instructions for Render

This application is configured for deployment on Render.com with automated CI/CD from GitHub.

### Automated Deployment from GitHub

When you push to the main branch, the application will automatically deploy to Render. This is configured using GitHub Actions.

To set up automated deployments:

1. Go to your GitHub repository settings
2. Add the following secrets:
   - `RENDER_SERVICE_ID`: Your Render service ID (found in your service URL or settings)
   - `RENDER_API_KEY`: Your Render API key (generated in Render dashboard)

### Deployment Configuration

- The `render.yaml` file defines the web service and database configuration
- `render-build.sh` is executed during the build phase
- `render-start.sh` is executed to start the application
- `.github/workflows/render-deploy.yml` handles the GitHub Actions workflow
- `Dockerfile` and `docker-compose.yml` provide containerization support

### Manual Deployment Steps

1. Log in to Render.com
2. Create a new Web Service
3. Connect your repository
4. Use the following settings:
   - **Build Command**: `./render-build.sh`
   - **Start Command**: `./render-start.sh`
   - **Environment Variables**:
     - `NODE_ENV`: `production`
     - `PORT`: Leave this to Render (it will be set automatically)
     - `DATABASE_URL`: Connect to your Render PostgreSQL database

### Database Setup

1. Create a PostgreSQL database in Render
2. Link it to your web service by setting the `DATABASE_URL` environment variable
3. The application will automatically create the required tables on first run

## Local Development

### Using NPM

To run the application locally:

```bash
npm run dev
```

This will start both the backend and frontend development servers.

### Using Docker

You can also run the application using Docker:

```bash
docker compose up
```

This will start the application and a PostgreSQL database in containers.