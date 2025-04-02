#!/bin/bash

# Cross-platform startup script for PanicSense
# Author: Mark Andrei R. Castillo

echo "🚀 Starting PanicSense..."

# Detect platform
if [ -n "$REPL_ID" ]; then
  PLATFORM="replit"
elif [ -n "$RENDER" ]; then
  PLATFORM="render"
else
  PLATFORM="local"
fi

echo "🔍 Detected platform: $PLATFORM"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
  echo "📝 Creating .env file from example..."
  cp .env.example .env
fi

# Run the compatibility check script
echo "✅ Running cross-platform compatibility check..."
node scripts/ensure-compatibility.js

# Run database migrations
echo "🗃️ Running database migrations..."
node migrations/run-migrations.js

# Platform-specific startup
if [ "$PLATFORM" = "replit" ]; then
  echo "🔄 Starting application in Replit environment..."
  npm run dev
elif [ "$PLATFORM" = "render" ]; then
  echo "🌐 Starting application in Render environment..."
  npm start
else
  echo "💻 Starting application in local environment..."
  npm run dev
fi