# Docker Setup for PanicSense

This guide provides instructions for running PanicSense using Docker for both local development and production deployment.

## Prerequisites

- Docker and Docker Compose installed on your system
- Git installed (for cloning the repository)
- Groq API keys for sentiment analysis

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/draiimon/PanicSense.git
cd PanicSense
```

### 2. Configure Environment Variables

Copy the example environment file and add your Groq API keys:

```bash
cp .env.example .env
```

Edit the `.env` file to add your Groq API keys:

```
GROQ_API_KEY_1=your_first_groq_api_key_here
VALIDATION_API_KEY=your_validation_api_key_here
```

### 3. Start the Docker Environment

```bash
docker-compose up --build
```

This will:
- Build the Docker containers
- Install all necessary dependencies
- Start the application in development mode
- Mount your local code for live reloading

The application will be available at http://localhost:5000

### 4. Development Workflow

- The code is mounted into the container, so changes to the code will be reflected immediately
- The application uses Vite's hot module replacement for quick frontend updates
- Database changes are managed through Drizzle ORM

## Production Deployment on Render

### 1. Fork or Clone the Repository

First, ensure you have a copy of the repository in your own GitHub account.

### 2. Connect to Render

1. Create an account on [Render](https://render.com) if you don't have one
2. In your Dashboard, click "New" and select "Blueprint"
3. Connect your GitHub account and select your forked/cloned repository
4. Render will automatically detect the `render.yaml` configuration

### 3. Configure Environment Variables

During the setup process, you'll need to add the following secrets:
- `GROQ_API_KEY_1` - Your primary Groq API key
- `VALIDATION_API_KEY` - Your validation Groq API key
- Any additional API keys as needed

The database will be automatically configured using the information in the `render.yaml` file.

### 4. Deploy

Complete the setup process, and Render will automatically deploy your application.

## Docker Commands Reference

### View Application Logs

```bash
docker-compose logs -f app
```

### Restart the Application

```bash
docker-compose restart app
```

### Stop All Containers

```bash
docker-compose down
```

### Rebuild and Start (after making changes to Dockerfile)

```bash
docker-compose up --build
```

## Troubleshooting

### Database Connection Issues

- Check if the `DATABASE_URL` environment variable is correctly set
- Ensure your IP is allowed in the Neon database's connection policies

### Container Won't Start

- Check logs: `docker-compose logs app`
- Ensure all required environment variables are set
- Verify you have the correct Docker and Docker Compose versions

### API Errors

- Verify your Groq API keys are correctly set in the environment variables
- Check the API request limits for your Groq account
- Look for errors in the application logs

## Additional Configuration

For advanced configuration options and deployment scenarios, refer to:
- [Dockerfile](../Dockerfile) for the container configuration
- [docker-compose.yml](../docker-compose.yml) for the local development setup
- [render.yaml](../render.yaml) for the Render deployment configuration

## Support

For additional assistance, refer to the main [README.md](../README.md) or contact the project maintainers.