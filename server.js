/**
 * MINIMAL EXPRESS SERVER FOR RENDER DEPLOYMENT
 * This file has ZERO TypeScript dependencies and NO top-level await
 */

const express = require('express');
const path = require('path');
const { Pool } = require('pg');
const http = require('http');
const { WebSocketServer } = require('ws');

// Create app and server
const app = express();
const server = http.createServer(app);
const port = process.env.PORT || 10000;

// Middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// WebSocket setup
const wss = new WebSocketServer({ server, path: '/ws' });
wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  ws.send(JSON.stringify({ type: 'connection', message: 'Connected to server' }));
  ws.on('close', () => console.log('WebSocket client disconnected'));
});

// Broadcast function for WebSockets
function broadcastUpdate(data) {
  wss.clients.forEach((client) => {
    if (client.readyState === 1) {
      client.send(JSON.stringify({
        ...data,
        timestamp: new Date().toISOString()
      }));
    }
  });
}

// Setup database connection
let pool;
if (process.env.DATABASE_URL) {
  console.log('🔌 Connecting to PostgreSQL database...');
  pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
  });
} else {
  console.warn('⚠️ No DATABASE_URL provided, database features will be disabled');
}

// Serve static files from client/dist
const distDir = path.join(__dirname, 'client/dist');
app.use(express.static(distDir, { maxAge: 31536000 }));

// Simple API routes
app.get('/api/health', async (req, res) => {
  try {
    if (pool) {
      const client = await pool.connect();
      await client.query('SELECT NOW()');
      client.release();
      res.json({
        status: 'ok',
        database: 'connected',
        timestamp: new Date().toISOString(),
        env: process.env.NODE_ENV || 'development',
        version: '1.0.0'
      });
    } else {
      res.json({
        status: 'ok',
        database: 'not configured',
        timestamp: new Date().toISOString(),
        env: process.env.NODE_ENV || 'development',
        version: '1.0.0'
      });
    }
  } catch (err) {
    console.error('Error in health check:', err);
    res.status(500).json({ status: 'error', message: err.message });
  }
});

// Simple echo route
app.get('/api/echo', (req, res) => {
  res.json({
    message: 'Server is running',
    timestamp: new Date().toISOString()
  });
});

// Catch-all route for SPA
app.get('*', (req, res) => {
  res.sendFile(path.join(distDir, 'index.html'));
});

// Start server
async function startServer() {
  try {
    console.log('========================================');
    console.log(`Starting server in ${process.env.NODE_ENV || 'development'} mode`);
    
    // Test database connection
    if (pool) {
      try {
        const client = await pool.connect();
        console.log('✅ Successfully connected to PostgreSQL database');
        client.release();
      } catch (err) {
        console.error('❌ Failed to connect to PostgreSQL database:', err);
      }
    }
    
    // Start listening
    server.listen(port, '0.0.0.0', () => {
      console.log(`========================================`);
      console.log(`🚀 Server running on port ${port}`);
      console.log(`Server ready at: ${new Date().toISOString()}`);
      console.log(`========================================`);
    });
  } catch (err) {
    console.error('FATAL ERROR during startup:', err);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('HTTP server closed');
    if (pool) {
      pool.end().then(() => process.exit(0)).catch(() => process.exit(1));
    } else {
      process.exit(0);
    }
  });
});

// Start without any top-level await
startServer().catch(err => {
  console.error('Error during startup:', err);
  // Still start the server even if there's an error
  server.listen(port, '0.0.0.0', () => {
    console.log(`Server running on port ${port} (startup error occurred)`);
  });
});