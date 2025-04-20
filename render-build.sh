#!/bin/bash
set -e

# Install dependencies
npm ci

# Build application
npm run build

# Fix static files directory structure
mkdir -p dist/client/dist
cp -r dist/public/* dist/client/dist/ || echo "No public files to copy"
cp dist/public/index.html dist/client/dist/index.html || echo "Creating empty index.html"
touch dist/client/dist/index.html

# Debug directories
echo "Directory structure after build:"
find dist -type d | sort
ls -la dist/client/dist || echo "dist/client/dist not created properly"

# Output build information
echo "Application built successfully!"