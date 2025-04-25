/**
 * Custom build script for Render deployment
 * This script handles the build process for Render without using vite directly
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

console.log('========================================');
console.log('PanicSense Render Build Script');
console.log('========================================');

// Ensure we're in production mode
process.env.NODE_ENV = 'production';

try {
  // Step 1: Install all dependencies including dev dependencies
  console.log('📦 Installing dependencies (including devDependencies)...');
  execSync('npm install --production=false', { stdio: 'inherit' });
  
  // Step 2: Run vite build using npx to ensure it uses the local installation
  console.log('🏗️ Building frontend using locally installed vite...');
  execSync('npx vite build', { stdio: 'inherit' });
  
  // Step 3: Build server files using esbuild
  console.log('🔨 Building server using esbuild...');
  execSync('npx esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist', { stdio: 'inherit' });
  
  // Step 4: Create directory structure for running
  console.log('📂 Setting up directory structure...');
  
  // Create a .env file if it doesn't exist
  if (!fs.existsSync('.env')) {
    fs.writeFileSync('.env', 'NODE_ENV=production\n');
    console.log('✅ Created .env file with production setting');
  }
  
  console.log('✅ Build completed successfully');
} catch (error) {
  console.error('❌ Build failed:', error.message);
  process.exit(1);
}