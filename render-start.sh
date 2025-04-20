#!/bin/bash
set -e

# Set environment variables
export NODE_ENV=production
export DEBUG=express:*

# Database setup
echo "Running database setup..."
echo "Database URL exists: $(if [ -n "$DATABASE_URL" ]; then echo "YES"; else echo "NO"; fi)"
npm run db:push || echo "Database setup skipped"

# Check for static files
echo "Checking for static files..."
if [ -d "dist/public" ]; then
  echo "✅ Static files directory exists"
  ls -la dist/public
else
  echo "❌ ERROR: Static files directory not found!"
  echo "Available directories:"
  ls -la
fi

# Start the server
echo "Starting server..."
node server-deploy.js