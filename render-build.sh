#!/bin/bash

# Render.com build script for PanicSense
set -e  # Exit on error

# Print environment information
echo "========== ENVIRONMENT INFO =========="
echo "Node version: $(node -v)"
echo "NPM version: $(npm -v)"
echo "Working directory: $(pwd)"

# Install ALL dependencies (including dev dependencies)
echo "========== INSTALLING DEPENDENCIES =========="
echo "Installing all dependencies (including dev dependencies)..."
npm install --production=false

# Ensure vite is available
echo "========== CHECKING VITE =========="
if ! command -v ./node_modules/.bin/vite &> /dev/null; then
    echo "Vite not found in node_modules! Installing explicitly..."
    npm install --no-save vite
    echo "Vite explicitly installed: $(./node_modules/.bin/vite --version)"
fi

# Build the frontend and server
echo "========== BUILDING APPLICATION =========="
echo "Building the frontend and server..."
npm run build

# Verify build output
echo "========== VERIFYING BUILD =========="
if [ -d "dist/public" ]; then
    echo "✅ Frontend build successful! Files found in dist/public"
    ls -la dist/public
else
    echo "⚠️ Warning: Frontend build folder not found!"
fi

echo "========== BUILD COMPLETE =========="
echo "Build completed successfully!"