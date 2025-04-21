#!/bin/bash
set -e

echo "✅ Preparing server environment..."

# Create necessary directories
mkdir -p server/public
mkdir -p logs
mkdir -p uploads

# Check for client build
if [ -d "client/dist" ]; then
  echo "📂 Using client/dist for static files"
  cp -r client/dist/* server/public/
fi

# Ensure DATABASE_URL has SSL mode if needed
if [ -n "$DATABASE_URL" ] && [ "$DB_SSL_REQUIRED" = "true" ]; then
  # Check if sslmode is already in the URL
  if test "$DATABASE_URL" != "${DATABASE_URL%\?sslmode=*}" || test "$DATABASE_URL" != "${DATABASE_URL%\&sslmode=*}"; then
    echo "🔒 SSL mode already present in DATABASE_URL"
  else
    export DATABASE_URL="${DATABASE_URL}?sslmode=require"
    echo "🔒 Added SSL mode to DATABASE_URL"
  fi
fi

echo "⚙️ Environment: NODE_ENV=$NODE_ENV"
echo "🔄 Attempting database connection..."

echo "🚀 Starting server on port $PORT..."

# Start the server using the primary server.js file
if [ -f "server.js" ]; then
  exec node server.js
else
  echo "❌ Error: server.js not found!"
  exit 1
fi