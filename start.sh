#!/bin/bash
set -e

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h db -U postgres; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done

echo "PostgreSQL is up - executing database migration"

# Run database migrations
echo "Pushing database schema changes..."
npm run db:push

# Set proper permissions for Python scripts
chmod +x python/process.py

# Run application in production mode
echo "Starting application in production mode..."
if [ "$NODE_ENV" = "production" ]; then
  node dist/index.js
else
  npm run dev
fi