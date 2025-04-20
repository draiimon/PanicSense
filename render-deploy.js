#!/usr/bin/env node

/**
 * This is a special deployment script for Render.com
 * It handles all the necessary steps to build the project correctly
 * for deployment without requiring package.json edits
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('Starting Render deployment process...');

// Step 1: Create production server file
try {
  console.log('Creating production server file...');
  const productionServer = `/**
 * This is a special production server file for Render deployment
 * It properly handles ESM modules and top-level await
 */

import express from 'express';
import { Server } from 'http';
import { WebSocketServer } from 'ws';
import { registerRoutes } from './routes.js';
import { simpleDbFix } from './db-simple-fix.js';
import { serveStatic } from './vite.js';

// Create Express server
const app = express();
let server;

// This function encapsulates all the async initialization
async function initializeServer() {
  console.log('========================================');
  console.log(\`Starting server initialization at: \${new Date().toISOString()}\`);
  console.log('========================================');

  try {
    // Apply database fixes for production
    console.log('Running database fixes...');
    await simpleDbFix();

    // Basic middleware
    app.use(express.json({ limit: '50mb' }));
    app.use(express.urlencoded({ extended: false, limit: '50mb' }));

    // Register routes
    server = await registerRoutes(app);
    console.log('Routes registered successfully');
    
    // Error handling middleware
    app.use((err, _req, res, _next) => {
      console.error('Server error:', err);
      res.status(500).json({ 
        error: true,
        message: err.message || 'Internal Server Error',
        timestamp: new Date().toISOString()
      });
    });
    
    // We're in production
    console.log('Running in production mode, serving static files...');
    serveStatic(app);
    
    // Get port from environment with fallback
    const port = parseInt(process.env.PORT || '10000', 10);
    console.log(\`Attempting to listen on port \${port}...\`);
    
    // Start server
    server.listen(port, '0.0.0.0', () => {
      console.log(\`========================================\`);
      console.log(\`🚀 Server running on port \${port}\`);
      console.log(\`Server listening at: http://0.0.0.0:\${port}\`);
      console.log(\`Server ready at: \${new Date().toISOString()}\`);
      console.log(\`========================================\`);
    });
    
    // Setup graceful shutdown
    process.on('SIGTERM', () => {
      console.log('SIGTERM received, shutting down gracefully');
      server?.close(() => {
        console.log('HTTP server closed');
        process.exit(0);
      });
      
      // Force close after 10 seconds
      setTimeout(() => {
        console.log('Forcing shutdown after timeout');
        process.exit(1);
      }, 10000);
    });
    
    process.on('SIGINT', () => {
      console.log('SIGINT received, shutting down gracefully');
      server?.close(() => {
        console.log('HTTP server closed');
        process.exit(0);
      });
      
      // Force close after 10 seconds
      setTimeout(() => {
        console.log('Forcing shutdown after timeout');
        process.exit(1);
      }, 10000);
    });
    
    return true;
  } catch (error) {
    console.error('Fatal error during server initialization:', error);
    return false;
  }
}

// Export the app and server for testing/monitoring
export { app, server };

// Start the server
console.log('Starting production server...');
initializeServer().catch(err => {
  console.error('Uncaught error during server initialization:', err);
  process.exit(1);
});
`;

  fs.writeFileSync(path.join(process.cwd(), 'server', 'production-server.ts'), productionServer);
  console.log('Production server file created successfully');
} catch (err) {
  console.error('Failed to create production server file:', err);
  process.exit(1);
}

// Step 2: Build the frontend
try {
  console.log('Building frontend...');
  execSync('npx vite build', { stdio: 'inherit' });
  console.log('Frontend build completed successfully');
} catch (err) {
  console.error('Frontend build failed:', err);
  process.exit(1);
}

// Step 3: Build the backend with the production server
try {
  console.log('Building backend...');
  execSync(
    'npx esbuild server/production-server.ts --platform=node --packages=external --bundle --format=esm --outfile=dist/index.js',
    { stdio: 'inherit' }
  );
  console.log('Backend build completed successfully');
} catch (err) {
  console.error('Backend build failed:', err);
  process.exit(1);
}

console.log('Render deployment build process completed successfully!');