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
  node -e "import('./server/db-simple-fix.js').then(module => module.simpleDbFix().then(result => console.log('Database fix result:', result)));"
fi

# Setup Python environment if needed (for AI analysis)
if [ -d "server/python" ]; then
  echo "Setting up Python environment..."
  # Use the specific PYTHONPATH for Render deployment
  export PYTHONPATH="/opt/render/project/src:/opt/render/project/src/server:/opt/render/project/src/server/python"
  echo "PYTHONPATH set to: $PYTHONPATH"
  
  # Install Python dependencies if needed
  if [ -f "server/python/requirements.txt" ]; then
    echo "Installing Python requirements from server/python/requirements.txt"
    pip install -r server/python/requirements.txt
  fi
fi

# Create uploads directory if it doesn't exist
mkdir -p uploads/temp
chmod -R 755 uploads

# Start the server with all necessary environment variables
echo "Starting main application server..."
NODE_ENV=production node dist/index.js