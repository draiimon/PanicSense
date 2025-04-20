# Docker & Render Deployment for PanicSense PH

This repository contains the Docker configuration and deployment setup for deploying the PanicSense PH application to Render.com.

## Docker Configuration

The Docker setup consists of:

1. **Dockerfile**: A multi-stage build that:
   - First stage: Builds the Node.js application
   - Second stage: Creates a production image with both Node.js and Python

2. **docker-verify.sh**: A verification script to locally test your Docker setup before deploying to Render.

3. **.dockerignore**: Specifies files and directories to exclude from the Docker image to keep it slim.

## Render Configuration

The Render deployment is configured with:

1. **render.yaml**: Defines the web service and database settings for Render Blueprint deployment.

2. **render-setup.sh**: A script that runs on startup in Render to set up the environment and run migrations.

3. **.env.example**: A template for required environment variables.

## Key Features

- **Multi-stage Docker build**: Efficiently creates a production-ready image
- **PostgreSQL database**: Automatically provisioned by Render
- **Python integration**: Includes Python 3.11 with PyTorch for ML features
- **Automatic migrations**: Database schema is updated during deployment
- **Environment variables**: Securely configured through Render

## Deployment Process

Detailed instructions for deploying to Render are provided in the [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md) file.

In summary:
1. Push your code to a Git repository
2. Connect your repository to Render
3. Render will detect your configuration and deploy automatically

## Local Development & Testing

To verify your Docker setup locally before deploying to Render:

```bash
# Make the verification script executable
chmod +x docker-verify.sh

# Run the verification script
./docker-verify.sh
```

This will:
1. Build your Docker image using the multi-stage Dockerfile
2. Start a temporary PostgreSQL container for testing
3. Run the application container and connect it to the database
4. Test the API health endpoint to ensure everything works
5. Clean up all test containers

If the verification passes, your setup is ready for deployment to Render!

## Technology Stack

- **Frontend**: React with TypeScript
- **Backend**: Node.js (Express)
- **Machine Learning**: Python with PyTorch
- **Database**: PostgreSQL
- **Deployment**: Docker on Render.com

## Directory Structure

- `/client`: React frontend
- `/server`: Node.js backend
- `/server/python`: Python ML components
- `/shared`: Shared code between frontend and backend
- `/migrations`: Database migrations
- `/public`: Static assets

## Additional Resources

- [Render.com Documentation](https://render.com/docs)
- [Docker Documentation](https://docs.docker.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)