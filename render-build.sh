#!/bin/bash
set -e

# Install dependencies
npm ci

# Build application
npm run build

# Fix static file permissions
chmod -R 755 dist/public

# Check if built files exist
echo "Verifying build output..."
if [ -d "dist/public" ]; then
  echo "✅ dist/public directory exists"
  ls -la dist/public
  
  if [ -f "dist/public/index.html" ]; then
    echo "✅ index.html exists"
  else
    echo "❌ ERROR: index.html not found!"
  fi
else
  echo "❌ ERROR: dist/public directory not found!"
  echo "Available directories:"
  ls -la
fi

# Output build information
echo "Application built successfully!"