/**
 * Enhanced Render.com deployment startup script for PanicSense
 * This simple starter loads the CommonJS version for compatibility
 */

// Import the CJS version directly for better compatibility with Render
import('./index.js').catch(err => {
  console.error('Failed to import server:', err);
  process.exit(1);
});