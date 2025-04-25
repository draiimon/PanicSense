/**
 * Render.com compatible server for PanicSense
 * Purpose: Provide a minimal server that works reliably on Render with development mode
 */

const express = require('express');
const path = require('path');
const fs = require('fs');
const http = require('http');

// Use development mode as requested
process.env.NODE_ENV = 'development';
const PORT = process.env.PORT || 10000;

// Display startup information
console.log('========================================');
console.log(`🚀 STARTING PANICSENSE (RENDER) ON PORT ${PORT}`);
console.log(`📅 Time: ${new Date().toISOString()}`);
console.log(`🌍 NODE_ENV: ${process.env.NODE_ENV}`);
console.log('========================================');

// Create Express application
const app = express();
const server = http.createServer(app);

// Configure middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Health check API endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    version: '1.0.0',
    time: new Date().toISOString(),
    env: process.env.NODE_ENV,
    note: 'Development mode for Render deployment'
  });
});

// Serve static files from the built frontend
const distPath = path.join(__dirname, 'dist', 'public');
if (fs.existsSync(path.join(distPath, 'index.html'))) {
  console.log(`✅ Found frontend files in: ${distPath}`);
  app.use(express.static(distPath));
} else {
  console.error('❌ WARNING: No frontend files found!');
}

// Enable CORS for development
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  next();
});

// Handle all other routes for SPA
app.get('*', (req, res) => {
  // Skip API routes that weren't matched
  if (req.path.startsWith('/api/')) {
    return res.status(404).json({ error: 'API endpoint not found' });
  }
  
  // For all other routes, serve the SPA's index.html
  const indexPath = path.join(distPath, 'index.html');
  if (fs.existsSync(indexPath)) {
    res.sendFile(indexPath);
  } else {
    res.status(404).send('Frontend not found. Build may be incomplete.');
  }
});

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log('========================================');
  console.log(`🚀 DEVELOPMENT SERVER RUNNING AT: http://0.0.0.0:${PORT}`);
  console.log(`📅 Server ready at: ${new Date().toISOString()}`);
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