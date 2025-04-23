/**
 * PanicSense Free Tier Render Deployment Start Script
 * 
 * This file provides a simple, reliable entry point for Render.com free tier deployment.
 * It forwards execution to the main server/index-wrapper.js file.
 */

console.log('=== 🚀 PanicSense Startup - Free Tier Render Edition ===');

// Set production environment
process.env.NODE_ENV = 'production';

// Import and run the actual server
try {
  console.log('Starting PanicSense server...');
  require('./server/index-wrapper.js');
  console.log('Server loaded successfully!');
} catch (error) {
  console.error('Error starting server:', error);
  console.error('Stack trace:', error.stack);
  process.exit(1);
}