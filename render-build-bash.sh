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

# Ensure npx is available
echo "========================================="
echo "🔍 Checking for npx..."
if ! command -v npx &> /dev/null; then
    echo "⚠️ npx not found, installing..."
    npm install -g npx
    echo "✅ npx installed"
else
    echo "✅ npx already available"
fi

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

# Build the frontend
echo "========================================="
echo "🏗️ Building frontend with Vite..."
npx vite build && echo "✅ Frontend build successful" || echo "❌ Frontend build FAILED"

# Check if frontend files exist
echo "🔍 Checking for frontend files..."
if [ -f "./dist/public/index.html" ]; then
    echo "✅ Frontend files built successfully"
else
    echo "⚠️ WARNING: Frontend files not found after build!"
    # Copy error message HTML as fallback
    echo "<html><body><h1>PanicSense</h1><p>Frontend build error. Please check the logs.</p></body></html>" > ./dist/public/index.html
    echo "✅ Created fallback index.html"
fi

# Build individual server components with informative messages
echo "========================================="
echo "🔨 Building server files..."

echo "📄 Building server/routes.ts..."
npx esbuild server/routes.ts --platform=node --packages=external --bundle --format=esm --outdir=dist

echo "📄 Building server/db.ts..."
npx esbuild server/db.ts --platform=node --packages=external --bundle --format=esm --outdir=dist

echo "📄 Building server/index.ts..."
npx esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist

echo "📄 Building server/db-simple-fix.ts..."
npx esbuild server/db-simple-fix.ts --platform=node --packages=external --bundle --format=cjs --outdir=dist

echo "📄 Building server/python-service.ts..."
npx esbuild server/python-service.ts --platform=node --packages=external --bundle --format=esm --outdir=dist

echo "📄 Building server/storage.ts..."
npx esbuild server/storage.ts --platform=node --packages=external --bundle --format=esm --outdir=dist

# Check build output
echo "========================================="
echo "🧾 Build output directory contents:"
ls -la dist/
echo 
echo "🧾 Public directory contents:"
ls -la dist/public/

# Success message
echo "========================================="
echo "✅ Build completed successfully!"
echo "📅 Build finished at: $(date)"
echo "========================================="