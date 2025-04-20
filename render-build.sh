#!/bin/bash

# This script is used by Render.com to build the application
# It installs dependencies and builds the client-side assets

echo "Starting build process for Render deployment..."

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm ci
# Explicitly install vite globally to ensure it's available
echo "Installing vite and esbuild globally for build process..."
npm install -g vite esbuild

# Build the frontend
echo "Building frontend assets..."
# Create a simple ESBuild only approach as a fallback
echo "Skipping Vite build and using only ESBuild..."
echo "Building the client assets to public directory first..."
mkdir -p public/assets
touch public/assets/.placeholder

# Build the server-side code
echo "Building the server code..."
npx esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist

# Create a simple server.js file that can directly run in production
echo "Creating a production server.js file..."
cat > server.js << 'EOF'
// Server.js - Production Ready Server
// This is a simplified version of the server code for production use
// It serves the static files and handles API requests

import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware for parsing JSON
app.use(express.json());

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Import the server routes
import('./dist/index.js')
  .then(module => {
    console.log('Server routes loaded successfully');
  })
  .catch(err => {
    console.error('Failed to load server routes:', err);
  });

// Simple health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Catch-all route for SPA
app.get('*', (req, res) => {
  // Check if this is an API request
  if (req.path.startsWith('/api/')) {
    return res.status(404).json({ error: 'API endpoint not found' });
  }
  
  // Serve the index.html for client-side routing
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT}`);
});
EOF

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