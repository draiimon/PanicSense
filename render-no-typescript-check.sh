#!/bin/bash
set -e

# Skip TypeScript checks and go straight to build
# This file is used by the render.yaml deployment

echo "Starting build with TypeScript checks disabled..."

# Install dependencies
npm ci

# Run the build without TypeScript checks
npx vite build
npx esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist

echo "Build completed successfully without TypeScript checks!"