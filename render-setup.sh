#!/bin/bash

# Simple Render deployment setup script
echo "=== 🚀 PanicSense Render Setup ==="

# Create necessary directories
mkdir -p uploads/{temp,data,profile_images}

# Install Python dependencies
if [ -f "render-requirements.txt" ]; then
  pip install -r render-requirements.txt
fi

# Build the frontend
npm run build

echo "=== ✅ Setup complete ==="