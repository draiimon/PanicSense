/**
 * Main entry point for the application on Render.com
 * This is a pure JavaScript file with no top-level await
 */

import express from 'express';
import { createServer } from 'http';
import path from 'path';
import { WebSocketServer } from 'ws';
import { fileURLToPath } from 'url';
import pg from 'pg';

// Get directory path
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const distDir = path.join(__dirname, 'client/dist');

// Create Express app
const app = express();
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Create server
const server = createServer(app);

// Create WebSocket server
const wss = new WebSocketServer({ server });
wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  ws.on('close', () => console.log('WebSocket client disconnected'));
});

// Connect to PostgreSQL database
let pool;
if (process.env.DATABASE_URL) {
  pool = new pg.Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
  });
}

// Initialize database
async function initDatabase() {
  if (!pool) {
    console.warn('No DATABASE_URL provided, skipping database initialization');
    return;
  }
  
  try {
    const client = await pool.connect();
    console.log('✅ Successfully connected to PostgreSQL database');
    client.release();
  } catch (err) {
    console.error('❌ Failed to connect to PostgreSQL database:', err);
  }
}

// Setup static file serving and routes
app.use(express.static(distDir));

// Health check route
app.get('/api/health', async (req, res) => {
  try {
    if (pool) {
      const client = await pool.connect();
      await client.query('SELECT NOW()');
      client.release();
      res.json({ status: 'ok', database: 'connected', timestamp: new Date().toISOString() });
    } else {
      res.json({ status: 'ok', database: 'not configured', timestamp: new Date().toISOString() });
    }
  } catch (err) {
    res.status(500).json({ status: 'error', message: err.message });
  }
});

// Fallback route for SPA
app.get('*', (req, res) => {
  res.sendFile(path.join(distDir, 'index.html'));
});

// Start the server (no top-level await)
const port = process.env.PORT || 10000;

// Initialize database and start server without top-level await
initDatabase().then(() => {
  server.listen(port, '0.0.0.0', () => {
    console.log(`========================================`);
    console.log(`Server running on port ${port}`);
    console.log(`Server listening at: http://0.0.0.0:${port}`);
    console.log(`Server ready at: ${new Date().toISOString()}`);
    console.log(`========================================`);
  });
}).catch(err => {
  console.error('Error during startup:', err);
  // Still start the server even if database fails
  server.listen(port, '0.0.0.0', () => {
    console.log(`Server running on port ${port} (database connection failed)`);
  });
});

// Handle graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('HTTP server closed');
    if (pool) {
      pool.end().then(() => {
        console.log('Database pool closed');
        process.exit(0);
      });
    } else {
      process.exit(0);
    }
  });
});