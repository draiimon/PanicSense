#!/bin/bash

# Ultra-simple Render deployment setup script
echo "=== 🚀 PanicSense Render Setup ==="

# Create necessary directories for uploads and data
mkdir -p uploads/{temp,data,profile_images}

# Install Python dependencies
if [ -f "render-requirements.txt" ]; then
  pip install -r render-requirements.txt
fi

# No build needed - we'll just run the server directly
echo "Skip build - using server directly"

echo "=== ✅ Setup complete ==="