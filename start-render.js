/**
 * Simple startup script for Render.com deployment
 */

console.log('Starting PanicSense on Render...');

// Set production mode explicitly
process.env.NODE_ENV = 'production';

// Simply require server index
require('./server/index-wrapper.js');