#!/bin/bash

# Simple build script for Render.com deployment
# This avoids any issues with top-level await by only building the frontend
# and using our plain JavaScript production server

echo "Starting simplified build process for Render.com deployment..."

# Make sure node_modules are installed
echo "Installing dependencies..."
npm install

# Build the frontend only
echo "Building frontend..."
npx vite build

# Make sure our production server is executable and properly formatted
echo "Ensuring production server setup is correct..."
chmod +x server/production-server.js

# Create an empty placeholder dist directory
# This is only needed because Render will look for it in the default start command
echo "Creating placeholder dist directory..."
mkdir -p dist
echo '// This is a placeholder file. The actual server is in server/production-server.js' > dist/placeholder.js

# Create a short note explaining we're NOT using TypeScript compilation
echo "Creating deployment note..."
echo "/*
IMPORTANT: This deployment uses server/production-server.js directly
and bypasses TypeScript compilation to avoid top-level await issues.
*/" > dist/README.txt

# Make sure our start script is executable
chmod +x render-start.sh

# Success message
echo "Build completed successfully!"
echo "The deployment will use server/production-server.js directly."