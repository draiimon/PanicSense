/**
 * Production server file
 * This is a JavaScript file (not TypeScript) to avoid any compilation issues
 * It's designed to run directly on Render.com
 */

import express from 'express';
import { createServer } from 'http';
import path from 'path';
import { WebSocketServer } from 'ws';
import { fileURLToPath } from 'url';
import pg from 'pg';

// Environment check
if (!process.env.DATABASE_URL) {
  console.error('DATABASE_URL environment variable is required');
  process.exit(1);
}

// Get directory paths using ESM compatible methods
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const publicDir = path.join(__dirname, '../client/dist');

// Setup database connection
const { Pool } = pg;
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
});

// Test database connection
(async () => {
  try {
    const client = await pool.connect();
    console.log('✅ Successfully connected to PostgreSQL database');
    client.release();
  } catch (err) {
    console.error('❌ Failed to connect to PostgreSQL database:', err);
  }
})();

// Simple database initialization to ensure tables exist
async function initDatabase() {
  try {
    const client = await pool.connect();
    try {
      // Check if users table exists
      const tableExists = await client.query(`
        SELECT EXISTS (
          SELECT FROM information_schema.tables 
          WHERE table_schema = 'public'
          AND table_name = 'users'
        );
      `);
      
      if (!tableExists.rows[0].exists) {
        console.log('Tables do not exist, initializing database...');
        // For simplicity, we're not recreating all tables here
        // Just acknowledging the condition
        console.log('Database needs initialization, please use migrations tool');
      } else {
        console.log('Database tables already exist, skipping initialization');
      }
    } finally {
      client.release();
    }
  } catch (err) {
    console.error('Error during database initialization:', err);
  }
}

// Create Express app
const app = express();

// Basic middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: false, limit: '50mb' }));

// Create HTTP server
const server = createServer(app);

// Create WebSocket server
const wss = new WebSocketServer({ server });
wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  ws.on('close', () => console.log('WebSocket client disconnected'));
});

// Setup static file serving for the frontend
app.use(express.static(publicDir));

// API routes
app.get('/api/health', async (req, res) => {
  try {
    // Test database connection
    const client = await pool.connect();
    await client.query('SELECT NOW()');
    client.release();
    
    res.json({ 
      status: 'ok', 
      timestamp: new Date().toISOString(),
      database: 'connected'
    });
  } catch (err) {
    res.status(500).json({
      status: 'error',
      message: 'Database connection failed',
      error: err.message
    });
  }
});

// Fallback to serve index.html for client-side routing
app.get('*', (req, res) => {
  res.sendFile(path.join(publicDir, 'index.html'));
});

// Initialize database and then start server
initDatabase().then(() => {
  const port = process.env.PORT || 10000;
  server.listen(port, '0.0.0.0', () => {
    console.log(`========================================`);
    console.log(`🚀 Production server running on port ${port}`);
    console.log(`Server listening at: http://0.0.0.0:${port}`);
    console.log(`Server ready at: ${new Date().toISOString()}`);
    console.log(`========================================`);
  });
}).catch(err => {
  console.error('Failed to initialize database:', err);
  process.exit(1);
});

// Handle graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('HTTP server closed');
    pool.end().then(() => {
      console.log('Database connections closed');
      process.exit(0);
    });
  });
  
  // Force exit after 10 seconds
  setTimeout(() => {
    console.log('Forcing shutdown after timeout');
    process.exit(1);
  }, 10000);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  server.close(() => {
    console.log('HTTP server closed');
    pool.end().then(() => {
      console.log('Database connections closed');
      process.exit(0);
    });
  });
  
  // Force exit after 10 seconds
  setTimeout(() => {
    console.log('Forcing shutdown after timeout');
    process.exit(1);
  }, 10000);
});