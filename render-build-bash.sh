#!/bin/bash

# Enhanced Render Build Script for PanicSense with full debugging
# This script properly builds both frontend and API server

echo "========================================="
echo "🚀 PANICSENSE RENDER BUILD SCRIPT - FULL DEBUG MODE"
echo "========================================="
echo "📅 Build started at: $(date)"
echo "📂 Current directory: $(pwd)"
echo "🧾 Directory contents:"
ls -la

# Install ALL dependencies including devDependencies
echo "========================================="
echo "📦 Installing ALL dependencies (including devDependencies)..."
npm install --production=false
echo "✅ Dependencies installed"

# Create necessary directories
echo "========================================="
echo "📁 Creating necessary directories..."
mkdir -p dist/public
mkdir -p uploads
echo "✅ Directories created"

# Show Node.js and npm versions
echo "========================================="
echo "🔢 Node.js version: $(node -v)"
echo "🔢 npm version: $(npm -v)"
echo "========================================="

# Build the frontend with a special command that skips Vite ESM issues
echo "========================================="
echo "🏗️ Building frontend with Vite..."
NODE_OPTIONS="--no-warnings" npm run build
echo "✅ Frontend build completed"

# Build CommonJS versions of server files
echo "========================================="
echo "📄 Building server files with ESBuild..."
npx esbuild server/db.ts --platform=node --packages=external --bundle --format=cjs --outdir=dist
npx esbuild server/index.ts --platform=node --packages=external --bundle --format=cjs --outdir=dist
echo "✅ Server build completed"

# Copy CommonJS files for maximum compatibility
echo "========================================="
echo "📄 Copying CommonJS files for maximum compatibility..."
# Copy the db-simple-fix.cjs file
cp -f server/db-simple-fix.cjs dist/db-simple-fix.cjs
# Copy the routes.js file as routes.cjs
cp -f server/routes.js dist/routes.cjs 2>/dev/null || echo "No routes.js found, skipping"
echo "✅ CommonJS files copied"

# Copy the start-render.cjs file
echo "========================================="
echo "📄 Copying start-render.cjs file..."
cp -f start-render.cjs dist/start-render.cjs
echo "✅ Start script copied"

# Check what has been built
echo "========================================="
echo "🧾 Build output directory contents:"
ls -la dist
echo "🧾 Public directory contents:"
ls -la dist/public
echo "========================================="
echo "✅ Build completed successfully!"
echo "📅 Build finished at: $(date)"
echo "========================================="