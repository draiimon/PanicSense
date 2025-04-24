#!/bin/bash

# Render deployment setup script
echo "=== 🚀 Render Setup Script ==="
echo "Current directory: $(pwd)"
echo "Node version: $(node -v)"

# Create necessary directories
mkdir -p dist/public uploads/{temp,data,profile_images} python

# Copy frontend files if they exist
if [ -d "client/dist" ]; then
  echo "Copying client/dist to dist/public..."
  cp -r client/dist/* dist/public/ 2>/dev/null || echo "Warning: client/dist files not copied"
fi

# Copy Python files
if [ -d "python" ]; then
  echo "Python directory already exists"
else
  echo "Creating python directory and copying files..."
  mkdir -p python
  
  # Copy Python files from their location
  if [ -d "server/python" ]; then
    cp -r server/python/* python/ 2>/dev/null || echo "Warning: Python files not copied"
  elif [ -d "python" ]; then
    cp -r python/* python/ 2>/dev/null || echo "Warning: Python files not copied"
  fi
fi

echo "=== ✅ Render setup complete ==="