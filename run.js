/**
 * PanicSense Runner - Single Entry Point for Render.com
 * Handles all necessary setup and starts the application
 * 
 * Usage on Render.com:
 * - Build Command: npm install
 * - Start Command: node run.js
 */

// Run setup first
require('./render-setup.js');

// Then start the main application
require('./index.js');