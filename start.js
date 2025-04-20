#!/usr/bin/env node

/**
 * DEPLOYMENT STARTER FOR RENDER.COM
 * 
 * This file completely bypasses the TypeScript build system
 * It does not use any of the TypeScript compiled files
 * Instead, it directly starts our production JavaScript server
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration
const PORT = process.env.PORT || 10000;
const ENV = process.env.NODE_ENV || 'production';

// Print some debug info
console.log(`Node version: ${process.version}`);
console.log(`Starting in ${ENV} mode on port ${PORT}`);

// Check if our index.js file exists
const indexPath = path.join(__dirname, 'index.js');
if (!fs.existsSync(indexPath)) {
  console.error(`ERROR: Cannot find ${indexPath}`);
  process.exit(1);
}

// Start the server using child_process (avoids any top-level await issues completely)
console.log(`Starting server: node ${indexPath}`);
const server = spawn('node', [indexPath], {
  env: { ...process.env, PORT, NODE_ENV: ENV },
  stdio: 'inherit'
});

// Handle server process exit
server.on('exit', (code) => {
  console.log(`Server process exited with code ${code}`);
  process.exit(code);
});

// Handle signals for graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down server');
  server.kill('SIGTERM');
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down server');
  server.kill('SIGINT');
});