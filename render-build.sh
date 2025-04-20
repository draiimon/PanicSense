#!/bin/bash

# This script is used by Render.com to build the application
# It installs dependencies and builds the client-side assets

echo "Starting build process for Render deployment..."

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm ci

# Build the frontend
echo "Building frontend assets..."
npm run build

# Install Python dependencies if needed
if [ -d "python" ]; then
  echo "Setting up Python environment and installing dependencies..."
  # Install Python packages
  if [ -f "python/requirements.txt" ]; then
    pip install -r python/requirements.txt
  fi
  
  # Install NLP dependencies if needed
  if [ -f "python/nlp_setup.py" ]; then
    python python/nlp_setup.py
  fi
  
  # Download NLTK data if needed
  if [ -f "python/nltk_setup.py" ]; then
    python python/nltk_setup.py
  fi
fi

# Make scripts executable
chmod +x render-start.sh

echo "Setting up file permissions..."
# Setup any other required permissions
if [ -d "uploads" ]; then
  chmod -R 755 uploads
fi
if [ -d "temp" ]; then
  chmod -R 755 temp
fi

echo "Build process completed successfully. Ready for deployment."