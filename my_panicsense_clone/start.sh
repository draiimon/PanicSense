#!/bin/bash
set -e

echo "📦 Installing all dependencies..."
npm install --no-audit --no-fund --prefer-offline

echo "🚧 Building client + bundling server..."
npm run build    # this runs: `vite build && esbuild server/index.ts ...`

echo "✅ Build complete. Starting app..."
exec node dist/index.js
