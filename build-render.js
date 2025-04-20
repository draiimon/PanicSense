#!/usr/bin/env node

// Build script for Render deployment
// This handles building both the frontend and backend for production

const { execSync } = require('child_process');

// First build the frontend
console.log('Building frontend...');
try {
  execSync('vite build', { stdio: 'inherit' });
} catch (e) {
  console.error('Frontend build failed:', e);
  process.exit(1);
}

// Then build the backend
console.log('Building backend...');
try {
  execSync('esbuild server/production.ts --platform=node --packages=external --bundle --format=esm --outfile=dist/index.js', { stdio: 'inherit' });
} catch (e) {
  console.error('Backend build failed:', e);
  process.exit(1);
}

console.log('Build completed successfully');