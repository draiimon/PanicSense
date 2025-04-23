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
      
      // Check multiple possible locations for frontend files
      const distPath = path.join(process.cwd(), 'dist', 'public');
      const clientDistPath = path.join(process.cwd(), 'client', 'dist');
      const publicPath = path.join(process.cwd(), 'public');
      const clientPublicPath = path.join(process.cwd(), 'client', 'public');
      
      // Try to run render-setup.sh if it exists and we can't find frontend files
      const hasRenderSetup = fs.existsSync('./render-setup.sh');
      if (hasRenderSetup && 
          !fs.existsSync(path.join(distPath, 'index.html')) && 
          !fs.existsSync(path.join(clientDistPath, 'index.html'))) {
        console.log('📋 Running render-setup.sh to prepare static files...');
        try {
          const { execSync } = require('child_process');
          execSync('chmod +x ./render-setup.sh && ./render-setup.sh', { stdio: 'inherit' });
          console.log('✅ render-setup.sh completed successfully');
        } catch (error) {
          console.error('⚠️ Error running render-setup.sh:', error.message);
        }
      }
      
      console.log('Checking for static files in multiple possible directories...');
      
      // Log the existence of directories for debugging
      console.log('Directory status:');
      console.log(`- dist/public: ${fs.existsSync(distPath) ? 'exists' : 'missing'}`);
      console.log(`- client/dist: ${fs.existsSync(clientDistPath) ? 'exists' : 'missing'}`);
      console.log(`- public: ${fs.existsSync(publicPath) ? 'exists' : 'missing'}`);
      console.log(`- client/public: ${fs.existsSync(clientPublicPath) ? 'exists' : 'missing'}`);
      
      // Try all possible frontend directories
      if (fs.existsSync(path.join(distPath, 'index.html'))) {
        console.log(`✅ Found frontend files in: ${distPath}`);
        app.use(express.static(distPath));
        app.use('*', (_req, res) => {
          res.sendFile(path.join(distPath, 'index.html'));
        });
      }
      else if (fs.existsSync(path.join(clientDistPath, 'index.html'))) {
        console.log(`✅ Found frontend files in: ${clientDistPath}`);
        app.use(express.static(clientDistPath));
        app.use('*', (_req, res) => {
          res.sendFile(path.join(clientDistPath, 'index.html'));
        });
      }
      else if (fs.existsSync(path.join(publicPath, 'index.html'))) {
        console.log(`✅ Found frontend files in: ${publicPath}`);
        app.use(express.static(publicPath));
        app.use('*', (_req, res) => {
          res.sendFile(path.join(publicPath, 'index.html'));
        });
      }
      else if (fs.existsSync(path.join(clientPublicPath, 'index.html'))) {
        console.log(`✅ Found frontend files in: ${clientPublicPath}`);
        app.use(express.static(clientPublicPath));
        app.use('*', (_req, res) => {
          res.sendFile(path.join(clientPublicPath, 'index.html'));
        });
      }
      else {
        console.log('⚠️ WARNING: No frontend files found in any directory!');
        
        // Emergency: Try to copy from client/build if it exists (for CRA projects)
        const clientBuildPath = path.join(process.cwd(), 'client', 'build');
        if (fs.existsSync(path.join(clientBuildPath, 'index.html'))) {
          console.log('🔥 Emergency: Found React build files in client/build, using those...');
          app.use(express.static(clientBuildPath));
          app.use('*', (_req, res) => {
            res.sendFile(path.join(clientBuildPath, 'index.html'));
          });
        } else {
          console.log('Server will run in API-only mode');
          
          // Provide a helpful error page
          app.get('/', (_req, res) => {
            res.status(200).send(`
              <!DOCTYPE html>
              <html lang="en">
              <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>PanicSense - API Only Mode</title>
                <style>
                  body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
                  header { background-color: #e74c3c; color: white; padding: 20px; border-radius: 5px; }
                  h1 { margin: 0; }
                  .content { margin-top: 20px; }
                  .api-box { background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #e74c3c; margin: 20px 0; }
                  footer { margin-top: 30px; font-size: 0.8em; color: #666; border-top: 1px solid #eee; padding-top: 10px; }
                </style>
              </head>
              <body>
                <header>
                  <h1>PanicSense API Server</h1>
                </header>
                <div class="content">
                  <p>The PanicSense backend API server is running correctly in API-only mode.</p>
                  <div class="api-box">
                    <h3>API Available</h3>
                    <p>The API endpoints are available at <code>/api/*</code></p>
                    <p>Key endpoints:</p>
                    <ul>
                      <li><code>/api/health</code> - Server health check</li>
                      <li><code>/api/sentiment-posts</code> - Get analyzed sentiment data</li>
                      <li><code>/api/disaster-events</code> - Get disaster events data</li>
                    </ul>
                  </div>
                </div>
                <footer>
                  <p>PanicSense Disaster Intelligence Platform</p>
                </footer>
              </body>
              </html>
            `);
          });
        }
      }
      
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