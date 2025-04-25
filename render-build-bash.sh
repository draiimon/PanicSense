#!/bin/bash

# Render Build Script for PanicSense
# This script fixes the Vite not found error

echo "========================================="
echo "PanicSense Render Build Script"
echo "========================================="

# Install ALL dependencies including devDependencies
echo "📦 Installing ALL dependencies (including devDependencies)..."
npm install --production=false

# Ensure npx is available
echo "🔍 Checking for npx..."
if ! command -v npx &> /dev/null; then
    echo "⚠️ npx not found, installing..."
    npm install -g npx
fi

# Build the frontend using npx vite
echo "🏗️ Building frontend with Vite..."
npx vite build

# Build the server 
echo "🔨 Building server with esbuild..."
npx esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist

# Success message
echo "✅ Build completed successfully!"
echo "========================================="