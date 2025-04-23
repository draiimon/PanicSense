/**
 * PanicSense Server - Production Ready
 */

// Environment setup
process.env.NODE_ENV = 'production';

// Show startup message
console.log('🚨 PanicSense Server Starting - Production Mode');

// Import the required modules
try {
  // Try direct import of compiled server if available
  if (require('fs').existsSync('./dist/index.js')) {
    console.log('Using compiled server at ./dist/index.js');
    require('./dist/index.js');
  } else {
    // Fall back to the wrapper
    console.log('Using wrapper at ./server/index-wrapper.js');
    require('./server/index-wrapper.js');
  }
} catch (err) {
  console.error('Error starting server:', err);
  
  // Try fallback options
  try {
    console.log('Trying fallback: server/index.js');
    require('./server/index.js');
  } catch (err2) {
    console.error('All server startup methods failed!');
    console.error(err2);
    process.exit(1);
  }
}