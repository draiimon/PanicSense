/**
 * Main server entry point - CommonJS version
 * For use in production deployment
 */

const express = require('express');
const http = require('http');
const path = require('path');

// Export for index-wrapper.js
function cleanupAndExit(server) {
  console.log('Cleaning up before exit...');
  if (server) {
    server.close(() => {
      console.log('Server closed, exiting now');
      process.exit(0);
    });
    
    // Force close after timeout if graceful close hangs
    setTimeout(() => {
      console.log('Could not close server gracefully, forcing exit');
      process.exit(1);
    }, 5000);
  } else {
    process.exit(0);
  }
}

// Export functions
module.exports = { cleanupAndExit };