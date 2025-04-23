/**
 * This is a special wrapper around the main index.ts file
 * It wraps all top-level await calls in an async IIFE
 * to ensure compatibility with CommonJS in production
 */

const express = require('express');
const session = require('express-session');
const path = require('path');
const { registerRoutes } = require('./routes');
const { simpleDbFix } = require('./db-simple-fix');
const { cleanupAndExit } = require('./index');

// Import log function that works without Vite
function defaultLog(message, source = "express") {
  const formattedTime = new Date().toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });
  console.log(`${formattedTime} [${source}] ${message}`);
}

// Declare the function types but initialize later
let log = defaultLog;

// Create Express server
const app = express();
let server;

// Export constants
const SERVER_START_TIMESTAMP = new Date().getTime();

// Wrap the server initialization in an async IIFE
(async () => {
  try {
    console.log('========================================');
    console.log(`Starting server initialization at: ${new Date().toISOString()}`);
    console.log('========================================');
    
    // Database initialization
    if (process.env.NODE_ENV !== 'production') {
      console.log('Running simple database fix in development...');
      await simpleDbFix();
    }

    // Session setup
    app.use(express.json());
    app.use(
      session({
        secret: process.env.SESSION_SECRET || 'keyboard cat',
        resave: false,
        saveUninitialized: true,
        cookie: { secure: false }
      })
    );
    
    // Create WebSocket server and register routes
    server = await registerRoutes(app);
    
    // Setup static file serving for production
    console.log(`Current NODE_ENV: ${process.env.NODE_ENV}`);
    
    if (process.env.NODE_ENV === 'production') {
      console.log('Running in production mode, serving static files...');
      // Simple implementation of serveStatic for production
      const distPath = path.join(process.cwd(), 'dist', 'public');
      console.log(`Serving static files from: ${distPath}`);
      app.use(express.static(distPath));
      app.use('*', (_req, res) => {
        res.sendFile(path.join(distPath, 'index.html'));
      });
      console.log('Static file serving setup complete');
    }
    
    // Error handling middleware
    app.use((err, _req, res, _next) => {
      console.error(err.stack);
      res.status(500).json({
        success: false,
        message: 'An internal server error occurred',
        error: process.env.NODE_ENV === 'production' ? undefined : err.message,
      });
    });
    
    // Listen on port
    const port = parseInt(process.env.PORT || '5000');
    console.log(`Attempting to listen on port ${port}...`);
    
    server.listen(port, '0.0.0.0', () => {
      console.log('========================================');
      console.log(`🚀 Server running on port ${port}`);
      console.log(`Server listening at: http://0.0.0.0:${port}`);
      console.log(`Server ready at: ${new Date().toISOString()}`);
      console.log('========================================');
    });
    
    // Handle graceful shutdown
    process.on('SIGINT', () => {
      console.log('Received SIGINT signal, shutting down gracefully...');
      cleanupAndExit(server);
    });
    
    process.on('SIGTERM', () => {
      console.log('Received SIGTERM signal, shutting down gracefully...');
      cleanupAndExit(server);
    });
    
  } catch (error) {
    console.error('Error during server initialization:', error);
    process.exit(1);
  }
})();

// Export the app and server for testing
module.exports = { app, server, SERVER_START_TIMESTAMP };