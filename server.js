/**
 * PRODUCTION SERVER FOR RENDER DEPLOYMENT
 * This file has ZERO TypeScript dependencies and NO top-level await
 * It implements the core functionality of the app for deployment
 */

const express = require('express');
const { Pool } = require('pg');
const http = require('http');
const path = require('path');
const fs = require('fs');
const { WebSocketServer } = require('ws');
const multer = require('multer');

// Create Express app and HTTP server
const app = express();
const server = http.createServer(app);
const port = process.env.PORT || 10000;

// Middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Setup file upload
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB limit
});

// Create WebSocket server
const wss = new WebSocketServer({ server, path: '/ws' });
wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  
  // Send initial message
  ws.send(JSON.stringify({
    type: 'connection_established',
    message: 'Connected to server',
    timestamp: Date.now()
  }));
  
  ws.on('close', () => console.log('WebSocket client disconnected'));
});

// Connect to PostgreSQL database
let pool;
if (process.env.DATABASE_URL) {
  console.log('Connecting to PostgreSQL database...');
  pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
  });
} else {
  console.warn('No DATABASE_URL provided, database features will be disabled');
}

// Create simple broadcast function for WebSocket
function broadcastUpdate(data) {
  wss.clients.forEach((client) => {
    if (client.readyState === 1) { // WebSocketServer.OPEN
      client.send(JSON.stringify({
        ...data,
        timestamp: Date.now()
      }));
    }
  });
}

// Register routes
async function registerRoutes() {
  // Basic health check
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
          mode: process.env.NODE_ENV || 'development'
        });
      } else {
        res.json({ 
          status: 'ok', 
          database: 'not configured', 
          timestamp: new Date().toISOString(),
          mode: process.env.NODE_ENV || 'development'
        });
      }
    } catch (err) {
      res.status(500).json({ status: 'error', message: err.message });
    }
  });

  // Echo route for testing
  app.get('/api/echo', (req, res) => {
    res.json({
      message: 'Server is running',
      timestamp: new Date().toISOString(),
      mode: process.env.NODE_ENV || 'development'
    });
  });

  // Core API routes (simplified for deployment)
  app.get('/api/disaster-events', async (req, res) => {
    try {
      if (!pool) return res.json([]);
      const result = await pool.query('SELECT * FROM disaster_events ORDER BY created_at DESC');
      res.json(result.rows);
    } catch (err) {
      console.error('Error getting disaster events:', err);
      res.status(500).json({ error: err.message });
    }
  });

  app.get('/api/sentiment-posts', async (req, res) => {
    try {
      if (!pool) return res.json([]);
      const result = await pool.query('SELECT * FROM sentiment_posts ORDER BY timestamp DESC');
      res.json(result.rows);
    } catch (err) {
      console.error('Error getting sentiment posts:', err);
      res.status(500).json({ error: err.message });
    }
  });

  app.get('/api/analyzed-files', async (req, res) => {
    try {
      if (!pool) return res.json([]);
      const result = await pool.query('SELECT * FROM analyzed_files ORDER BY created_at DESC');
      res.json(result.rows);
    } catch (err) {
      console.error('Error getting analyzed files:', err);
      res.status(500).json({ error: err.message });
    }
  });

  // Serve static files
  const distDir = path.join(__dirname, 'client/dist');
  if (fs.existsSync(distDir)) {
    app.use(express.static(distDir));
    console.log(`Serving static files from ${distDir}`);
    
    // Fallback for SPA routing
    app.get('*', (req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    console.warn(`Static directory ${distDir} not found`);
    
    // Simple fallback page
    app.get('*', (req, res) => {
      res.send(`
        <html>
          <head><title>Disaster Monitoring Platform</title></head>
          <body>
            <h1>Server is running but frontend is not built</h1>
            <p>API endpoints are available but the frontend was not properly built.</p>
          </body>
        </html>
      `);
    });
  }
  
  return server;
}

// Initialize database and start server
async function startServer() {
  try {
    console.log('========================================');
    console.log(`Starting server initialization at: ${new Date().toISOString()}`);
    console.log('========================================');
    
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
    
    // Register routes
    await registerRoutes();
    console.log('Routes registered successfully');

    // Start server
    server.listen(port, '0.0.0.0', () => {
      console.log(`========================================`);
      console.log(`🚀 Server running on port ${port}`);
      console.log(`Server listening at: http://0.0.0.0:${port}`);
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
      pool.end().then(() => {
        console.log('Database pool closed');
        process.exit(0);
      }).catch(() => process.exit(1));
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