#!/bin/bash

# This is a simplified startup script for Render.com deployment
# It completely bypasses the TypeScript compilation and top-level await issues
# by using our plain JavaScript production server directly

# Print versions for debugging
echo "Node.js version: $(node -v)"
echo "NPM version: $(npm -v)"

# Show warning if trying to use the old dist/index.js
if [ -f "dist/index.js" ]; then
  echo "WARNING: dist/index.js exists but will NOT be used!"
  echo "Using server/production-server.js instead to avoid await issues."
fi

# Make sure our production server file is executable
echo "Setting executable permissions for production server..."
chmod +x server/production-server.js

# Start the production server DIRECTLY
# This completely bypasses dist/index.js with its top-level await issues
echo "====================================================="
echo "STARTING PRODUCTION SERVER (server/production-server.js)"
echo "Bypassing TypeScript and top-level await completely"
echo "====================================================="

# The magic line that starts everything correctly
NODE_ENV=production PORT=$PORT node server/production-server.js