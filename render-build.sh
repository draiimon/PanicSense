#!/bin/bash

# This script is used by Render.com to build the application
# It installs dependencies and builds the client-side assets

echo "Starting build process for Render deployment..."

# Install dependencies
npm ci

# Build the frontend
npm run build

# Make the start script executable
chmod +x render-start.sh

echo "Build process completed successfully."