/**
 * Minimal Express server for Render deployment
 * This is a CommonJS fallback script for worst-case deployment scenarios
 */

const express = require('express');
const path = require('path');
const fs = require('fs');
const http = require('http');

// Use production mode
process.env.NODE_ENV = 'production';
const PORT = process.env.PORT || 10000;

// Log startup info
console.log('========================================');
console.log(`🚀 [RENDER] STARTING PANICSENSE EMERGENCY SERVER`);
console.log(`📅 Time: ${new Date().toISOString()}`);
console.log(`🔌 PORT: ${PORT}`);
console.log(`🌍 NODE_ENV: ${process.env.NODE_ENV}`);

// Create Express server
const app = express();
const server = http.createServer(app);

// Setup basic middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Basic health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    time: new Date().toISOString(),
    env: process.env.NODE_ENV
  });
});

// Serve static files from the dist/public directory
const distPath = path.join(__dirname, 'dist', 'public');
if (fs.existsSync(path.join(distPath, 'index.html'))) {
  console.log(`✅ Found frontend files in: ${distPath}`);
  app.use(express.static(distPath));
} else {
  console.error('❌ WARNING: No frontend files found!');
}

// Handle SPA routing
app.get('*', (req, res) => {
  // Skip API routes
  if (req.path.startsWith('/api/')) {
    return res.status(404).json({ error: 'API endpoint not found' });
  }
  
  // Serve index.html for client-side routing
  if (fs.existsSync(path.join(distPath, 'index.html'))) {
    res.sendFile(path.join(distPath, 'index.html'));
  } else {
    res.status(404).send('Frontend not found');
  }
});

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log('========================================');
  console.log(`🚀 SERVER RUNNING IN PRODUCTION MODE`);
  console.log(`📡 Server listening at: http://0.0.0.0:${PORT}`);
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