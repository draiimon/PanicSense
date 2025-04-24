/**
 * Simple startup script for Render.com deployment
 * This script uses CommonJS for maximum compatibility
 */

console.log('Starting PanicSense on Render...');

// Set production mode explicitly
process.env.NODE_ENV = 'production';

// Check if using Render
if (process.env.RENDER) {
  console.log('Detected Render environment');
}

// Import and run the main server
try {
  // Try the ESM version first
  import('./server/index.js')
    .catch(err => {
      console.log('Falling back to CommonJS version...');
      // Fall back to CommonJS version
      require('./index.js');
    });
} catch (err) {
  console.log('Falling back to main server entry...');
  require('./index.js');
}