#!/bin/bash

# SIMPLE BUILD SCRIPT FOR RENDER.COM
# This doesn't require any credit card info or blueprints
# Works with the FREE TIER!

echo "========== STARTING PANICSENSE BUILD =========="
echo "Node version: $(node -v)"
echo "NPM version: $(npm -v)"
echo "Current directory: $(pwd)"

# Install Node.js dependencies
echo "========== INSTALLING NODE.JS DEPENDENCIES =========="
npm install --include=dev 

# Build frontend
echo "========== BUILDING FRONTEND =========="
npm run build

# Create necessary directories
echo "========== CREATING REQUIRED DIRECTORIES =========="
mkdir -p dist
mkdir -p dist/public
mkdir -p dist/python
mkdir -p uploads

# Copy Python files
echo "========== PREPARING PYTHON SCRIPTS =========="
cp -r python/* dist/python/

# Make Python scripts executable
chmod +x python/*.py
chmod +x dist/python/*.py

# Make server scripts executable
chmod +x *.cjs
chmod +x *.js
chmod +x *.sh

# Create a marker file to help with debugging
echo "render-build.sh ran at $(date)" > build-timestamp.txt
cp build-timestamp.txt dist/

echo "========== BUILD COMPLETE =========="
echo "Build completed successfully!"