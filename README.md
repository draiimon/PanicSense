# PanicSense

A comprehensive real-time disaster news and sentiment analysis platform designed to provide critical emergency insights and community support across multiple regions.

## Key Features

- Advanced sentiment analysis engine with AI-powered insights
- TypeScript/React frontend with responsive design
- Multilingual support (English and Filipino)
- Groq AI for natural language processing
- Neon database for robust data persistence
- Multi-source disaster news aggregation and validation system

## Docker Setup

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Quick Start with Docker

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/panicsense.git
cd panicsense
```

2. **Create environment variables file**

```bash
cp .env.example .env
```

Edit the `.env` file to add your API keys and customize your configuration.

3. **Build and start the Docker containers**

```bash
docker-compose up -d
```

This will start:
- PanicSense application on port 5000
- PostgreSQL database on port 5432
- Redis cache on port 6379

4. **Access the application**

Open your browser and go to [http://localhost:5000](http://localhost:5000)

### Development Mode

To run in development mode with hot reloading:

```bash
# Update your .env file
BUILD_TARGET=builder
DEV_MODE=true

# Start the containers
docker-compose up -d
```

### Production Deployment

For production, you can use the following settings:

```bash
# In your .env file
NODE_ENV=production
BUILD_TARGET=runner
# Remove DEV_MODE or leave it empty

# Build and start the containers
docker-compose up -d --build
```

## Database Migrations

The application automatically handles database setup. When running for the first time or when schema changes, the container will:

1. Wait for the database to be ready
2. Run necessary setup scripts
3. Apply emergency fixes if needed

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| NODE_ENV | Environment (development/production) | production |
| PORT | Application port | 5000 |
| DATABASE_URL | PostgreSQL connection string | postgres://panicsense:panicsense@postgres:5432/panicsense |
| GROQ_API_KEY | Groq AI API key | (required) |
| DB_SSL_REQUIRED | Enable SSL for database connection | false |
| BUILD_TARGET | Docker build target | runner |
| DEV_MODE | Enable development mode with source code mounting | (empty) |

## Debugging

To view logs from the containers:

```bash
# All logs
docker-compose logs

# Application logs only
docker-compose logs app

# Follow logs in real-time
docker-compose logs -f app
```

## Cloud Deployment

For cloud deployment, you have several options:

### Using Docker Compose

1. Set up your cloud server with Docker and Docker Compose
2. Clone the repository and follow the same steps as local deployment
3. Make sure to use production settings

### Using Docker with External Database

1. Create a PostgreSQL database instance in your cloud provider
2. Update the `.env` file with the correct DATABASE_URL
3. Launch only the app container:

```bash
docker run -d -p 5000:80 \
  --env-file .env \
  --name panicsense \
  your-docker-repo/panicsense:latest
```

## Troubleshooting

### Database Connection Issues

If you have issues connecting to the database:

1. Check that your DATABASE_URL is correct
2. Ensure the database server is running and accessible
3. Try setting DB_SSL_REQUIRED=true if your cloud provider requires SSL

### Missing Dependencies

If the application reports missing dependencies:

1. Make sure you've built the Docker image correctly
2. Check that required Python libraries are installed
3. Verify that the entrypoint script is properly executing

For further assistance, check the logs or contact support.