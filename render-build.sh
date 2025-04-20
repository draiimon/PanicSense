#!/bin/bash

# This script is used by Render.com to build the application
# It installs dependencies and builds the client-side assets

echo "Starting build process for Render deployment..."

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm ci

# Build the frontend
echo "Building frontend assets..."
# Using npx to call vite and esbuild directly, avoiding the 'vite not found' error
npx vite build && npx esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist

# Install Python dependencies if needed
if [ -d "server/python" ]; then
  echo "Setting up Python environment and installing dependencies..."
  # Install Python packages
  if [ -f "server/python/requirements.txt" ]; then
    echo "Installing Python requirements from server/python/requirements.txt"
    pip install -r server/python/requirements.txt
  elif [ -f "python-requirements.txt" ]; then
    echo "Installing Python requirements from python-requirements.txt"
    pip install -r python-requirements.txt
  fi
  
  # Install NLTK data
  echo "Installing NLTK data..."
  python -m nltk.downloader punkt stopwords wordnet
fi

# Make scripts executable
chmod +x render-start.sh

echo "Setting up file permissions..."
# Create uploads directory if it doesn't exist
if [ ! -d "uploads" ]; then
  mkdir -p uploads/temp
fi
chmod -R 755 uploads

echo "Build process completed successfully. Ready for deployment."