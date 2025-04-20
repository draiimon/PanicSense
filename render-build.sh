#!/bin/bash
set -e

# Install dependencies
npm ci

# Build application
npm run build

# Output build information
echo "Application built successfully!"