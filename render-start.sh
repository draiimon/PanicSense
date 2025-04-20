#!/bin/bash

# This script is used by Render.com to start the application
# It checks the environment and starts the server

echo "Starting application in Render environment..."

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
  echo "WARNING: DATABASE_URL is not set. Database features will be disabled."
fi

# Apply any database fixes or migrations if needed
echo "Checking database schema and applying necessary fixes..."
if [ -f "server/db-simple-fix.js" ]; then
  # This will create any missing tables
  node -e "require('./server/db-simple-fix.js').simpleDbFix().then(result => console.log('Database fix result:', result));"
fi

# Setup Python environment if needed (for AI analysis)
if [ -d "python" ]; then
  echo "Setting up Python environment..."
  export PYTHONPATH=$PYTHONPATH:$(pwd)
  # Install Python dependencies if needed
  if [ -f "python/requirements.txt" ]; then
    pip install -r python/requirements.txt
  fi
fi

# Start the server with all necessary environment variables
echo "Starting main application server..."
NODE_ENV=production node server.js