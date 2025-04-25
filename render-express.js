/**
 * Render Express Server
 * A simple Express server for production that works on both Replit and Render
 * This avoids using Vite in production
 */

import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import fs from 'fs';
import http from 'http';
import { WebSocketServer } from 'ws';
import pg from 'pg';

// For ES modules compatibility
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const { Pool } = pg;

// Create Express app and HTTP server
const app = express();
const server = http.createServer(app);
const port = process.env.PORT || 5000;

// Middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Create WebSocket server for real-time updates
const wss = new WebSocketServer({ server, path: '/ws' });
wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  ws.send(JSON.stringify({
    type: 'connection_established',
    message: 'Connected to server',
    timestamp: Date.now()
  }));
  ws.on('close', () => console.log('WebSocket client disconnected'));
});

// Helper function for broadcast
function broadcastUpdate(data) {
  wss.clients.forEach((client) => {
    if (client.readyState === 1) {
      client.send(JSON.stringify({
        ...data,
        timestamp: Date.now()
      }));
    }
  });
}

// Database connection
let pool;
try {
  const databaseUrl = process.env.DATABASE_URL;
  if (databaseUrl) {
    pool = new Pool({
      connectionString: databaseUrl,
      ssl: { rejectUnauthorized: false }
    });
    console.log('Created database connection from DATABASE_URL');
  }
} catch (err) {
  console.error('Failed to create database connection:', err.message);
}

// API Routes
app.get('/api/health', async (req, res) => {
  try {
    if (pool) {
      const client = await pool.connect();
      await client.query('SELECT NOW()');
      client.release();
      res.json({ 
        status: 'ok', 
        database: 'connected', 
        timestamp: new Date().toISOString() 
      });
    } else {
      res.json({ 
        status: 'ok', 
        database: 'not configured', 
        timestamp: new Date().toISOString() 
      });
    }
  } catch (err) {
    res.status(500).json({ status: 'error', message: err.message });
  }
});

app.get('/api/disaster-events', async (req, res) => {
  try {
    if (!pool) return res.json([]);
    const result = await pool.query('SELECT * FROM disaster_events ORDER BY id DESC');
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
    const result = await pool.query('SELECT * FROM analyzed_files ORDER BY id DESC');
    res.json(result.rows);
  } catch (err) {
    console.error('Error getting analyzed files:', err);
    res.status(500).json({ error: err.message });
  }
});

// Serve static files - check multiple locations for compatibility with both Replit and Render
const possibleStaticPaths = [
  path.join(__dirname, 'dist/public'),
  path.join(__dirname, 'client/dist'),
  path.join(__dirname, 'public'),
  path.join(__dirname, 'client/public')
];

// Find the first valid static path
let staticPath = null;
for (const pathToCheck of possibleStaticPaths) {
  if (fs.existsSync(pathToCheck) && fs.existsSync(path.join(pathToCheck, 'index.html'))) {
    staticPath = pathToCheck;
    break;
  }
}

if (staticPath) {
  console.log(`Serving static files from ${staticPath}`);
  app.use(express.static(staticPath));
  
  // Fallback to index.html for SPA
  app.get('*', (req, res) => {
    if (req.path.startsWith('/api/')) {
      return res.status(404).json({ error: 'API endpoint not found' });
    }
    res.sendFile(path.join(staticPath, 'index.html'));
  });
} else {
  console.warn('No valid static path found');
  app.get('*', (req, res) => {
    if (req.path.startsWith('/api/')) {
      return res.status(404).json({ error: 'API endpoint not found' });
    }
    
    res.send(`
      <html>
        <head>
          <title>PanicSense</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
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

// Start server
server.listen(port, '0.0.0.0', () => {
  console.log(`========================================`);
  console.log(`🚀 Server running on port ${port}`);
  console.log(`Server listening at: http://0.0.0.0:${port}`);
  console.log(`Server ready at: ${new Date().toISOString()}`);
  console.log(`========================================`);
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
      }).catch(() => process.exit(1));
    } else {
      process.exit(0);
    }
  });
});