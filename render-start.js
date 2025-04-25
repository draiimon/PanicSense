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

// Direct approach - import the dist version
import express from 'express';
import session from 'express-session';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Import directly from dist
import { registerRoutes } from './dist/index.js';

// Create Express server
const app = express();

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

// Serve static files
const distPath = path.join(__dirname, 'dist', 'public');
if (fs.existsSync(path.join(distPath, 'index.html'))) {
  console.log(`✅ Found frontend files in: ${distPath}`);
  app.use(express.static(distPath));
  app.get('*', (req, res) => {
    // For API routes, continue to next handler
    if (req.path.startsWith('/api/')) {
      return;
    }
    res.sendFile(path.join(distPath, 'index.html'));
  });
} else {
  console.log('⚠️ WARNING: No frontend files found');
}

// Start the server
try {
  // Create WebSocket server and register routes
  const server = await registerRoutes(app);
  
  // Listen on port
  server.listen(PORT, '0.0.0.0', () => {
    console.log('========================================');
    console.log(`🚀 Server running on port ${PORT}`);
    console.log(`Server listening at: http://0.0.0.0:${PORT}`);
    console.log(`Server ready at: ${new Date().toISOString()}`);
    console.log('========================================');
  });
  
  // Handle graceful shutdown
  process.on('SIGINT', () => {
    console.log('Received SIGINT signal, shutting down gracefully...');
    server.close(() => {
      console.log('Server closed');
      process.exit(0);
    });
  });
  
  process.on('SIGTERM', () => {
    console.log('Received SIGTERM signal, shutting down gracefully...');
    server.close(() => {
      console.log('Server closed');
      process.exit(0);
    });
  });
} catch (error) {
  console.error('Failed to start server:', error);
  process.exit(1);
}