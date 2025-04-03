# PanicSense - Disaster Monitoring Platform

A sophisticated disaster monitoring platform for the Philippines that leverages advanced AI sentiment analysis to capture nuanced emotional responses during crisis situations.

## Deployment Instructions for Render

### Environment Variables Required

Set the following environment variables in your Render dashboard:

- `PORT`: 10000 (default)
- `NODE_ENV`: production
- `NODE_OPTIONS`: --max-old-space-size=512
- `DATABASE_URL`: Your PostgreSQL connection string

### Deployment Steps

1. Connect your GitHub repository to Render.
2. Create a new Web Service.
3. Select "Docker" as the environment.
4. Configure the environment variables listed above.
5. Deploy! Render will automatically use the Dockerfile in this repo.

## Local Development with Docker

To run this application locally using Docker:

```bash
# Build the Docker image
docker build -t panicsense .

# Run the container
docker run -p 10000:10000 \
  -e PORT=10000 \
  -e NODE_ENV=production \
  -e DATABASE_URL=your_database_url \
  panicsense
```

Alternatively, use Docker Compose:

```bash
# Create a .env file with your environment variables
echo "DATABASE_URL=your_database_url" > .env

# Run with Docker Compose
docker-compose up
```

## Key Technologies

- Node.js/React frontend with TypeScript
- PostgreSQL database with Drizzle ORM
- Advanced multi-level sentiment detection (panic, fear, resilience, neutral, disbelief)
- Context-aware AI sentiment analysis
- Multi-source social media event integration
- Machine learning predictive analytics for disaster response