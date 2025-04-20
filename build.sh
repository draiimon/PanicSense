#!/bin/bash

# Build script for Render.com deployment
# This is an alternative to render.yaml configuration

echo "Starting build process for Render deployment..."

# Execute our deployment script
node render-deploy.js

# Make sure the output directory exists
mkdir -p dist

# Success message
echo "Build process completed successfully!"