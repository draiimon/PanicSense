#!/bin/bash

# SIMPLE BUILD SCRIPT FOR RENDER.COM
# This doesn't require any credit card info or blueprints
# Works with the FREE TIER!

echo "========== STARTING PANICSENSE BUILD =========="
echo "Node version: $(node -v)"
echo "NPM version: $(npm -v)"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"

# Install Node.js dependencies
echo "========== INSTALLING NODE.JS DEPENDENCIES =========="
npm install

# Install Python dependencies
echo "========== INSTALLING PYTHON DEPENDENCIES =========="
pip install -r requirements.txt

# Create necessary directories
echo "========== CREATING REQUIRED DIRECTORIES =========="
mkdir -p dist
mkdir -p uploads

# Make Python scripts executable
echo "========== PREPARING PYTHON SCRIPTS =========="
chmod +x python/*.py

# Make server scripts executable
chmod +x *.cjs
chmod +x *.sh

# Create a marker file to help with debugging
echo "render-build.sh ran at $(date)" > build-timestamp.txt

echo "========== BUILD COMPLETE =========="
echo "Build completed successfully!"