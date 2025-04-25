/**
 * Simple startup script for Render.com deployment
 * This is a CommonJS version that avoids ESM-related issues on Render
 */

const express = require('express');
const session = require('express-session');
const path = require('path');
const fs = require('fs');
const http = require('http');

// Set environment variables
process.env.NODE_ENV = 'production';
const PORT = process.env.PORT || 10000;

// Log startup
console.log('========================================');
console.log(`Starting PanicSense in direct production mode (CJS)`);
console.log(`PORT: ${PORT}`);
console.log(`NODE_ENV: ${process.env.NODE_ENV}`);
console.log(`Start time: ${new Date().toISOString()}`);
console.log('========================================');

// Create Express server
const app = express();
const server = http.createServer(app);

// Setup middleware
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
} else {
  console.log('⚠️ WARNING: No frontend files found!');
}

// Define basic API routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', time: new Date().toISOString() });
});

// Catch-all route for SPA
app.get('*', (req, res) => {
  // Skip API routes
  if (req.path.startsWith('/api/')) {
    return;
  }
  
  // Serve the main index.html for all other routes (SPA)
  if (fs.existsSync(path.join(distPath, 'index.html'))) {
    res.sendFile(path.join(distPath, 'index.html'));
  } else {
    res.status(404).send('Not found');
  }
});

// Start the server
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