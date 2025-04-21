# PanicSense Docker Deployment Guide

This guide provides instructions for running the PanicSense application locally using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/PanicSense.git
   cd PanicSense
   ```

2. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file to configure your environment variables:
   - Set `SESSION_SECRET` to a strong random string
   - Add your `OPENAI_API_KEY` if you want to use AI features
   - Configure any other environment variables as needed

4. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

5. Access the application at http://localhost:3000

## Environment Variables

The following environment variables can be configured in your `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `NODE_ENV` | Application environment | `production` |
| `PORT` | Application port | `3000` |
| `DATABASE_URL` | PostgreSQL connection string | `postgres://postgres:postgres@db:5432/postgres` |
| `DB_SSL_REQUIRED` | Whether SSL is required for database connection | `false` |
| `SESSION_SECRET` | Secret for session encryption | (required) |
| `OPENAI_API_KEY` | OpenAI API key for AI features | (optional) |

## Development Mode

For development, you can use volume mounts to allow for code changes without rebuilding the container:

1. Uncomment the volume mounts in `docker-compose.yml`:
   ```yaml
   volumes:
     - ./client:/app/client
     - ./server:/app/server
     - ./python:/app/python
     - ./shared:/app/shared
   ```

2. Set `NODE_ENV=development` in your `.env` file

3. Rebuild and restart the containers:
   ```bash
   docker-compose down
   docker-compose up -d --build
   ```

## Troubleshooting

### Database Connection Issues

If you're experiencing database connection issues:

1. Check that the PostgreSQL container is running:
   ```bash
   docker-compose ps
   ```

2. Verify the database credentials in your `.env` file

3. You can connect to the PostgreSQL container to troubleshoot:
   ```bash
   docker-compose exec db psql -U postgres
   ```

### Python Module Issues

If Python processing is not working:

1. Check the logs for any Python-related errors:
   ```bash
   docker-compose logs app
   ```

2. You can enter the container to troubleshoot Python:
   ```bash
   docker-compose exec app bash
   cd python
   python3 process.py --help
   ```

## Database Management

### View Database Data

You can connect to the PostgreSQL database using:

```bash
docker-compose exec db psql -U postgres
```

Common PostgreSQL commands:
- `\dt` - List all tables
- `\d tablename` - Describe a table
- `SELECT * FROM tablename;` - View all records in a table

### Reset Database

To completely reset the database:

```bash
docker-compose down -v
docker-compose up -d
```

This will remove all volumes (including the database) and recreate them.