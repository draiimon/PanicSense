/**
 * Startup script for Render.com deployment
 * This script serves as an entry point for the application when deployed on Render
 */

// Set production environment
process.env.NODE_ENV = 'production';

// Set default port if not provided by Render
if (!process.env.PORT) {
  process.env.PORT = '10000';
}

console.log('========================================');
console.log(`Starting PanicSense server on Render`);
console.log(`NODE_ENV: ${process.env.NODE_ENV}`);
console.log(`PORT: ${process.env.PORT}`);
console.log('========================================');

// Determine which startup file to use based on available files
const fs = require('fs');
const path = require('path');

// Check for bundled server file first (preferred option)
if (fs.existsSync(path.join(__dirname, 'dist', 'index-wrapper.js'))) {
  console.log('Starting with bundled server (dist/index-wrapper.js)');
  // Use dynamic import for ESM compatibility
  import('./dist/index-wrapper.js')
    .catch(err => {
      console.error('Error starting bundled server:', err);
      process.exit(1);
    });
}
// Fallback to the wrapper script directly
else if (fs.existsSync(path.join(__dirname, 'server', 'index-wrapper.js'))) {
  console.log('Starting with server wrapper (server/index-wrapper.js)');
  // Use dynamic import for ESM compatibility
  import('./server/index-wrapper.js')
    .catch(err => {
      console.error('Error starting server wrapper:', err);
      process.exit(1);
    });
}
// Last resort - try the main index.ts with ts-node
else if (fs.existsSync(path.join(__dirname, 'server', 'index.ts'))) {
  console.log('Starting with server/index.ts (ts-node)');
  // Check if ts-node is available
  try {
    require('ts-node/register');
    require('./server/index.ts');
  } catch (err) {
    console.error('Error starting with ts-node:', err);
    console.log('Attempting to use tsx as a fallback...');
    
    // Try tsx as a fallback
    try {
      const { execSync } = require('child_process');
      execSync('npx tsx server/index.ts', { stdio: 'inherit' });
    } catch (tsxErr) {
      console.error('Error starting with tsx:', tsxErr);
      process.exit(1);
    }
  }
}
// No viable server entry point found
else {
  console.error('No server entry point found. Deployment failed.');
  process.exit(1);
}