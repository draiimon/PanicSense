#!/bin/bash

# This script is used by Render.com to start the application
# It checks the environment and starts the server

echo "Starting application in Render environment..."

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
  echo "WARNING: DATABASE_URL is not set. Database features will be disabled."
fi

# Start the server
node server.js