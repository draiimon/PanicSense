#!/bin/bash
set -e

echo "📦 Installing all dependencies..."
npm install --no-audit --no-fund --prefer-offline

echo "🚧 Building client + bundling server..."
npm run build

echo "✅ Build complete. Preparing server..."

# Make sure static files are properly staged
mkdir -p server/public
if [ -d "client/dist" ]; then
  cp -r client/dist/* server/public/
fi

echo "🚀 Starting server on port $PORT..."
exec node server.js