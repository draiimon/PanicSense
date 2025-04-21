#!/bin/bash
# Initialize the database and start the server

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
  echo "WARNING: DATABASE_URL environment variable is not set."
  echo "Will use default local PostgreSQL connection."
  export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/panicsense"
fi

# Ensure any required environment variables are set
if [ ! -f .env ]; then
  echo "Creating .env file from example..."
  cp .env.example .env
fi

# Wait for the database to be ready if needed
echo "Checking database connection..."
node verify-db.js || exit 1

# Ensure database schema is up to date
echo "Setting up database schema..."
npm run db:push

# Start the application server
echo "Starting PanicSense server..."
exec node server.js