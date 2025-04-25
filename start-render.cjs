#!/usr/bin/env node

/**
 * Simple starter script for PanicSense on Render.com (Free Tier)
 * This script runs production-server.cjs, which handles both Node.js and Python
 */

try {
  console.log('Starting PanicSense in production mode...');
  
  // Just execute the production server directly
  require('./production-server.cjs');
  
} catch (error) {
  console.error('Failed to start server:', error);
  process.exit(1);
}