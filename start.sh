#!/bin/bash
set -e

# Already built by Docker, no need to rebuild
echo "✅ Preparing server environment..."

# Create necessary directories
mkdir -p server/public
mkdir -p logs
mkdir -p uploads

# Check both possible locations for the client build
if [ -d "client/dist" ]; then
  echo "📂 Using client/dist for static files"
  cp -r client/dist/* server/public/
elif [ -d "dist/public" ]; then
  echo "📂 Using dist/public for static files"
  cp -r dist/public/* server/public/
fi

# Special handling for Render: If DATABASE_URL is defined but doesn't have SSL params, add them
if [ -n "$DATABASE_URL" ]; then
  # Check if sslmode is already in the URL
  if [[ "$DATABASE_URL" != *"?sslmode="* ]]; then
    export DATABASE_URL="${DATABASE_URL}?sslmode=require"
    echo "🔒 Added SSL mode to DATABASE_URL"
  fi
fi

echo "⚙️ Environment: NODE_ENV=$NODE_ENV"
echo "🔄 Attempting database connection..."

# If we're on Render, run special setup script
if [ "$RUNTIME_ENV" = "render" ] && [ -f "render-setup.js" ]; then
  echo "🚀 Running Render.com special setup script..."
  node render-setup.js
fi

echo "🚀 Starting server on port $PORT..."

# Use the right entry point based on which file exists
if [ -f "server.js" ]; then
  exec node server.js
elif [ -f "dist/index.js" ]; then
  exec node dist/index.js
else
  echo "❌ Error: No server entry point found!"
  exit 1
fi