#!/bin/bash

set -e

echo "🚧 Building client app..."
cd client
npm install
npm run build
cd ..

echo "📦 Installing server dependencies..."
cd server
npm install
cd ..

echo "📁 Copying built files to server/public..."
mkdir -p server/public
cp -r client/dist/* server/public/

echo "✅ Static files ready."
echo "🚀 Starting server on port $PORT..."
exec node server.js
