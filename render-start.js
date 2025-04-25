/**
 * Custom startup script for Render.com deployment
 * This ensures proper environment setup for production
 */

// Set production mode explicitly
process.env.NODE_ENV = 'production';

// Make sure we're using dynamic port assignment from Render
const PORT = process.env.PORT || 10000;
process.env.PORT = PORT;

console.log(`========================================`);
console.log(`Starting PanicSense on Render.com`);
console.log(`PORT: ${PORT}`);
console.log(`NODE_ENV: ${process.env.NODE_ENV}`);
console.log(`Start time: ${new Date().toISOString()}`);
console.log(`========================================`);

// Import the server wrapper
import('./server/index-wrapper.js').catch(err => {
  console.error('Failed to start server:', err);
  process.exit(1);
});