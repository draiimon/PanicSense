#!/bin/bash
set -e

# 1) Build the React client
cd client
npm ci --no-audit --no-fund --prefer-offline
npm run build
cd ..

echo "✅ Client build complete."

# 2) Install server deps
cd server
npm ci --no-audit --no-fund --prefer-offline
cd ..

echo "✅ Server dependencies installed."

# 3) Stage static files for Express
mkdir -p server/public
cp -r client/dist/* server/public/

echo "✅ Static assets staged."

# 4) Launch the server
echo "🚀 Starting server on port $PORT..."
exec node server/server.js