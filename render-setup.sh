#!/bin/bash
# Render setup script for PanicSense
# This script is executed during Render deployment to prepare the environment

echo "Starting Render deployment setup..."

# Load the render configuration using dynamic import (ESM compatible)
CONFIG=$(node -e "import('./render.config.js').then(m => console.log(JSON.stringify(m.default))).catch(e => console.error(e))")

if [ -z "$CONFIG" ]; then
  echo "Failed to load config, using default build command..."
  BUILD_COMMAND="npx next build && npx esbuild server/index-wrapper.js --platform=node --packages=external --bundle --format=esm --outdir=dist"
else
  # Extract build command from config
  BUILD_COMMAND=$(echo $CONFIG | jq -r '.buildCommand')
fi

echo "Running build command: $BUILD_COMMAND"
eval $BUILD_COMMAND

echo "Build process completed"

# Create a .env file if it doesn't exist
if [ ! -f .env ]; then
  echo "Creating .env file..."
  touch .env
  echo "NODE_ENV=production" >> .env
  echo "PORT=10000" >> .env
fi

# Ensure the dist directory exists
mkdir -p dist/public

# Copy any static files if they exist
if [ -d "public" ]; then
  echo "Copying public files to dist/public..."
  cp -r public/* dist/public/
fi

# Copy the Next.js output if it exists
if [ -d ".next" ]; then
  echo "Copying Next.js build output to dist/.next..."
  mkdir -p dist/.next
  cp -r .next/* dist/.next/
fi

# Create compatibility files for both ESM and CommonJS environments
echo "Creating compatibility wrappers for both ESM and CommonJS..."

# Create server starter for Render 
cat > dist/render-start.js << EOF
// ESM compatible server starter for Render
import('../start-render.js').catch(err => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
EOF

# Create next.config.cjs for compatibility
cat > next.config.cjs << EOF
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  distDir: 'dist/.next',
  experimental: {
    outputFileTracingRoot: process.cwd(),
  },
  publicRuntimeConfig: {
    apiBase: process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5000',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: '/api/:path*',
      },
      {
        source: '/ws',
        destination: '/ws',
      }
    ];
  },
};

module.exports = nextConfig;
EOF

echo "Making scripts executable..."
chmod +x start-render.js

echo "Render deployment setup complete!"