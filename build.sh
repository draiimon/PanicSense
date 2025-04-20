#!/bin/bash
# This script creates a CommonJS compatible version of dist/index.js

echo "=========================================="
echo "Creating CommonJS compatible dist/index.js"
echo "=========================================="

# Make sure dist directory exists
mkdir -p dist

# Check if the original index.js exists
if [ -f "./index.js" ]; then
  # Copy our pure CommonJS version to the dist folder
  cp ./index.js ./dist/index.js
  echo "✅ Successfully copied CommonJS version to dist/index.js"
else
  echo "❌ ERROR: index.js not found!"
  exit 1
fi

# Modify package.json to remove type:module
echo "Removing type:module from package.json"
node render-update-package.cjs

# Make sure the script is executable
chmod +x index.js

echo "=========================================="
echo "Build script completed successfully"
echo "=========================================="