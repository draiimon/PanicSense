#!/bin/bash
# Simple deployment script for Render
# Avoids Vite for better compatibility

echo "Starting PanicSense deployment on Render..."

# Set environment variables
export NODE_ENV=production
export PORT=${PORT:-10000}

# Ensure the dist directory exists
mkdir -p dist/public

# Build the client-side 
echo "Building client-side application..."
node build-static.js

# Build the server-side
echo "Building server-side application..."
npx esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist

echo "Deployment build complete!"

# Start the server
echo "Starting server with render-express..."
node render-express.js