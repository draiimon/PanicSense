#!/bin/bash

# Simple build script for Render.com deployment
# This avoids any issues with top-level await by only building the frontend

echo "Starting simplified build process for Render.com deployment..."

# Make sure node_modules are installed
echo "Installing dependencies..."
npm install

# Build the frontend only
echo "Building frontend..."
npx vite build

# Create the dist directory if it doesn't exist
mkdir -p dist

# Success message
echo "Build completed successfully!"
chmod +x render-start.sh