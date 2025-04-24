#!/bin/bash

# Render deployment setup script for PanicSense
echo "=== 🚀 PanicSense Render Setup Script ==="
echo "Current directory: $(pwd)"
echo "Node version: $(node -v)"

# Create necessary directories for uploads and temporary data
mkdir -p uploads/{temp,data,profile_images} python

# Copy Python files if not already in place
if [ -d "python" ]; then
  echo "Python directory exists"
else
  if [ -d "server/python" ]; then
    echo "Copying Python files from server/python..."
    cp -r server/python/* python/ 2>/dev/null || echo "Warning: Python files not copied"
  fi
fi

# Install Python requirements if running on Render
if [ -n "$RENDER" ] && [ -f "render-requirements.txt" ]; then
  echo "Installing Python requirements for Render..."
  pip install -r render-requirements.txt
fi

# Build the client if needed
if [ ! -d "client/dist" ] || [ -z "$(ls -A client/dist 2>/dev/null)" ]; then
  echo "Building client..."
  # Use the existing build script instead of build:client
  npm run build
else
  echo "Client build exists, skipping build step"
fi

# Ensure permissions are set correctly
chmod -R 755 uploads python

echo "=== ✅ Render setup complete ==="