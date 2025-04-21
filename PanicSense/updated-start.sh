#!/bin/bash
set -e

echo "🚧 [start.sh] Updating and verifying npm..."
npm install -g npm@latest

echo "📦 [start.sh] Installing build dependencies..."
npm install -g vite esbuild typescript

echo "🚧 [start.sh] Building client app..."
cd client
npm ci
npm run build
cd ..

echo "📦 [start.sh] Installing server dependencies..."
cd server
npm ci
cd ..

echo "📁 [start.sh] Staging static files..."
mkdir -p server/public
cp -r client/dist/* server/public/

echo "✅ [start.sh] Static assets are ready."
echo "🚀 [start.sh] Launching server on port $PORT..."
exec node server.js