/**
 * Main server for PanicSense web frontend
 * This works with the Python worker via a shared database
 */

const express = require('express');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { Pool } = require('pg');
const WebSocket = require('ws');

// Configure environment
const PORT = process.env.PORT || 10000;
const DATABASE_URL = process.env.DATABASE_URL;

// Create database pool
const pool = new Pool({
  connectionString: DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

// Create Express app and HTTP server
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Set up WebSocket for live updates
wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  
  // Periodically send event updates
  const intervalId = setInterval(async () => {
    try {
      const client = await pool.connect();
      const result = await client.query('SELECT * FROM disaster_events ORDER BY created_at DESC LIMIT 5');
      client.release();
      
      ws.send(JSON.stringify({
        type: 'disaster_update',
        data: result.rows,
        timestamp: new Date().toISOString()
      }));
    } catch (error) {
      console.error('WebSocket data fetch error:', error);
      // Send fallback data if database isn't available
      ws.send(JSON.stringify({
        type: 'disaster_update',
        data: [],
        timestamp: new Date().toISOString(),
        error: 'Database connection error'
      }));
    }
  }, 10000);
  
  ws.on('close', () => {
    console.log('WebSocket client disconnected');
    clearInterval(intervalId);
  });
});

// Configure middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Enable CORS
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  next();
});

// Health endpoint
app.get('/api/health', async (req, res) => {
  let dbStatus = 'unknown';
  
  try {
    const client = await pool.connect();
    await client.query('SELECT NOW() as now');
    client.release();
    dbStatus = 'connected';
  } catch (error) {
    dbStatus = 'error: ' + error.message;
  }
  
  res.json({
    status: 'ok',
    version: '1.0.0',
    time: new Date().toISOString(),
    env: process.env.NODE_ENV || 'development',
    database: dbStatus,
    pythonServiceUrl: process.env.PYTHON_SERVICE_URL || 'Not configured'
  });
});

// Connect to the Python service for analysis
app.post('/api/analyze', async (req, res) => {
  try {
    const { text } = req.body;
    
    if (!text) {
      return res.status(400).json({ error: 'Text is required' });
    }
    
    // Store analysis request in database for Python worker to pick up
    const client = await pool.connect();
    const result = await client.query(
      'INSERT INTO analysis_requests (text, status) VALUES ($1, $2) RETURNING id',
      [text, 'pending']
    );
    client.release();
    
    const requestId = result.rows[0].id;
    
    res.json({
      success: true,
      requestId,
      message: 'Analysis request submitted, check status endpoint'
    });
  } catch (error) {
    console.error('Analysis request error:', error);
    res.status(500).json({ error: 'Failed to submit analysis request' });
  }
});

// Get analysis results
app.get('/api/analysis/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const client = await pool.connect();
    const result = await client.query(
      'SELECT * FROM analysis_requests WHERE id = $1',
      [id]
    );
    client.release();
    
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Analysis request not found' });
    }
    
    const analysis = result.rows[0];
    
    res.json({
      id: analysis.id,
      text: analysis.text,
      status: analysis.status,
      results: analysis.results,
      created_at: analysis.created_at,
      completed_at: analysis.completed_at
    });
  } catch (error) {
    console.error('Analysis fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch analysis results' });
  }
});

// API endpoints that fetch from database
app.get('/api/disaster-events', async (req, res) => {
  try {
    const client = await pool.connect();
    const result = await client.query('SELECT * FROM disaster_events ORDER BY created_at DESC');
    client.release();
    res.json(result.rows);
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({ error: 'Failed to fetch disaster events' });
  }
});

app.get('/api/sentiment-posts', async (req, res) => {
  try {
    const client = await pool.connect();
    const result = await client.query('SELECT * FROM sentiment_posts ORDER BY timestamp DESC LIMIT 100');
    client.release();
    res.json(result.rows);
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({ error: 'Failed to fetch sentiment posts' });
  }
});

app.get('/api/analyzed-files', async (req, res) => {
  try {
    const client = await pool.connect();
    const result = await client.query('SELECT * FROM analyzed_files ORDER BY created_at DESC');
    client.release();
    res.json(result.rows);
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({ error: 'Failed to fetch analyzed files' });
  }
});

// Serve static files
const distPath = path.join(__dirname, 'dist', 'public');
if (fs.existsSync(path.join(distPath, 'index.html'))) {
  console.log(`✅ Found frontend files in: ${distPath}`);
  app.use(express.static(distPath));
} else {
  console.error(`❌ Frontend files not found at ${distPath}`);
}

// Catch-all route
app.get('*', (req, res) => {
  if (req.path.startsWith('/api/')) {
    return res.status(404).json({ error: 'API endpoint not found' });
  }
  
  const indexPath = path.join(distPath, 'index.html');
  if (fs.existsSync(indexPath)) {
    res.sendFile(indexPath);
  } else {
    res.status(404).send('Frontend not found');
  }
});

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 PanicSense Web Server running on http://0.0.0.0:${PORT}`);
  console.log(`📅 Started at: ${new Date().toISOString()}`);
});

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('Received SIGINT signal, shutting down gracefully');
  server.close(() => {
    pool.end();
    process.exit(0);
  });
});

process.on('SIGTERM', () => {
  console.log('Received SIGTERM signal, shutting down gracefully');
  server.close(() => {
    pool.end();
    process.exit(0);
  });
});