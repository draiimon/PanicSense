#!/bin/bash
# Simple setup script for Render.com

# Install Python dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
  echo "Installing Python dependencies..."
  pip install -r requirements.txt
fi

# Create uploads directory
mkdir -p uploads

echo "Setup complete!"