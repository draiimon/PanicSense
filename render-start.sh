#!/bin/bash
set -e

# Set environment variables
export NODE_ENV=production

# Run database migrations if needed
echo "Running database setup..."
npm run db:push || echo "Database setup skipped"

# Start the server
echo "Starting server..."
node server-deploy.js