#!/bin/bash

# Render deployment setup script
echo "=== 🚀 Render Setup Script ==="
echo "Current directory: $(pwd)"
echo "Node version: $(node -v)"

# Create dist if it doesn't exist
mkdir -p dist/public

# Copy frontend files if they exist
if [ -d "client/dist" ]; then
  echo "Copying client/dist to dist/public..."
  cp -r client/dist/* dist/public/ 2>/dev/null || true
elif [ -d "public" ]; then
  echo "Copying public to dist/public..."
  cp -r public/* dist/public/ 2>/dev/null || true
fi

# Create other necessary directories
echo "Creating directories..."
mkdir -p uploads/{temp,data,profile_images} python

# Copy Python files if needed
if [ -d "server/python" ]; then
  echo "Copying server/python to python..."
  cp -r server/python/* python/ 2>/dev/null || true
fi

echo "=== ✅ Render setup complete ==="