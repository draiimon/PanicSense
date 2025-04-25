#!/bin/bash

# Install dependencies including dev dependencies
echo "Installing all dependencies..."
npm install --production=false

# Run the build
echo "Building the application..."
npm run build

echo "Build completed!"