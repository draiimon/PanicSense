#!/bin/bash
set -e

# Already built by Docker, no need to rebuild
echo "✅ Preparing server environment..."

# Make sure static files are properly staged
mkdir -p server/public
if [ -d "client/dist" ]; then
  cp -r client/dist/* server/public/
fi

echo "🚀 Starting server on port $PORT..."
exec node server.js