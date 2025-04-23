/**
 * PanicSense Server - Production Ready
 * CommonJS version for complete reliability
 */

// Environment setup
process.env.NODE_ENV = 'production';

// Show startup message
console.log('🚨 PanicSense Server Starting - Production Mode');

// Import the required modules
try {
  // Try direct import of compiled server if available
  if (require('fs').existsSync('./dist/server/index.js')) {
    console.log('Using compiled server at ./dist/server/index.js');
    require('./dist/server/index.js');
  } else if (require('fs').existsSync('./server/index-wrapper.js')) {
    // Fall back to the wrapper
    console.log('Using wrapper at ./server/index-wrapper.js');
    require('./server/index-wrapper.js');
  } else {
    // Try another fallback
    console.log('Trying fallback: server/index.js');
    require('./server/index.js');
  }
} catch (err) {
  console.error('Error starting server:', err);
  
  // Show error details
  console.error('All server startup methods failed!');
  console.error(err);
  process.exit(1);
}