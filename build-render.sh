#!/bin/bash

# Install all dependencies including dev dependencies
echo "Installing all dependencies (including dev dependencies)..."
npm install --production=false

# Run the build
echo "Building the application..."
npm run build

# Success message
echo "Build completed successfully!"