#!/bin/bash

# This is a simplified startup script for Render.com deployment
# It's designed to be as simple as possible to avoid deployment issues

# Print Node.js version for debugging
echo "Node.js version: $(node -v)"
echo "NPM version: $(npm -v)"

# Make sure our server file is executable
chmod +x server/production-server.js

# Start the server using the plain JavaScript file
# This avoids any TypeScript compilation or top-level await issues
echo "Starting production server..."
NODE_ENV=production PORT=$PORT node server/production-server.js