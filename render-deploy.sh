#!/bin/bash

# This is the main deployment script for Render.com
# It completely avoids using TypeScript and the dist directory

echo "======================================================"
echo "RENDER DEPLOYMENT SCRIPT"
echo "======================================================"

# Make sure node_modules are installed
echo "Installing dependencies..."
npm install

# Build the frontend only
echo "Building frontend with Vite..."
npx vite build

# Make our start.js executable
echo "Setting permissions for start.js..."
chmod +x start.js

# Special handling for package.json to bypass dist/index.js
echo "Creating special version of package.json for deployment..."
node -e "
const fs = require('fs');
const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
pkg.scripts.start = 'node start.js';
pkg.main = 'start.js';
pkg.originalStart = pkg.scripts.start;
pkg._modified = 'Modified for Render deployment';
fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2));
"

# Test that our start.js is working
echo "Checking that start.js is valid JavaScript..."
node --check start.js

# Make sure our index.js is valid
echo "Checking that index.js is valid JavaScript..."
node --check index.js

# Finish with success message
echo "======================================================"
echo "DEPLOYMENT BUILD COMPLETE"
echo "The application will start using start.js"
echo "======================================================"