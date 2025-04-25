/**
 * EXTREMELY SIMPLE SERVER FOR RENDER.COM DEPLOYMENT
 * This file contains all needed code to run the PanicSense API server
 * without any complex build steps
 */

const express = require('express');
const { Pool } = require('pg');
const path = require('path');
const fs = require('fs');
const http = require('http');

// Create Express app
const app = express();
const port = process.env.PORT || 10000;
const server = http.createServer(app);

// Basic middleware
app.use(express.json());
app.use(express.static('public'));

// Database connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

// Test database connection
async function testDatabaseConnection() {
  try {
    const client = await pool.connect();
    const result = await client.query('SELECT NOW()');
    client.release();
    
    console.log('✅ Database connected successfully!');
    console.log(`✅ Server time: ${result.rows[0].now}`);
    return true;
  } catch (err) {
    console.error('❌ Database connection error:', err);
    return false;
  }
}

// API Routes
app.get('/api/disaster-events', async (req, res) => {
  try {
    const { rows } = await pool.query('SELECT * FROM disaster_events ORDER BY id DESC');
    res.json(rows);
  } catch (error) {
    console.error('Error fetching disaster events:', error);
    
    // Try with basic columns if error related to column names
    if (error.message.includes('column') && error.message.includes('not exist')) {
      try {
        const { rows } = await pool.query('SELECT id, name, description FROM disaster_events ORDER BY id DESC');
        return res.json(rows);
      } catch (fallbackError) {
        return res.status(500).json({ error: 'Database error', details: fallbackError.message });
      }
    }
    
    res.status(500).json({ error: 'Database error', details: error.message });
  }
});

app.get('/api/sentiment-posts', async (req, res) => {
  try {
    const { rows } = await pool.query('SELECT * FROM sentiment_posts ORDER BY id DESC LIMIT 100');
    res.json(rows);
  } catch (error) {
    console.error('Error fetching sentiment posts:', error);
    
    // Try with basic columns if error related to column names
    if (error.message.includes('column') && error.message.includes('not exist')) {
      try {
        const { rows } = await pool.query('SELECT id, text, sentiment, confidence FROM sentiment_posts ORDER BY id DESC LIMIT 100');
        return res.json(rows);
      } catch (fallbackError) {
        return res.status(500).json({ error: 'Database error', details: fallbackError.message });
      }
    }
    
    res.status(500).json({ error: 'Database error', details: error.message });
  }
});

app.get('/api/analyzed-files', async (req, res) => {
  try {
    const { rows } = await pool.query('SELECT * FROM analyzed_files ORDER BY id DESC');
    res.json(rows);
  } catch (error) {
    console.error('Error fetching analyzed files:', error);
    
    // Try with basic columns if error related to column names
    if (error.message.includes('column') && error.message.includes('not exist')) {
      try {
        const { rows } = await pool.query('SELECT id, original_name, stored_name FROM analyzed_files ORDER BY id DESC');
        return res.json(rows);
      } catch (fallbackError) {
        return res.status(500).json({ error: 'Database error', details: fallbackError.message });
      }
    }
    
    res.status(500).json({ error: 'Database error', details: error.message });
  }
});

app.get('/api/active-upload-session', async (req, res) => {
  try {
    const { rows } = await pool.query('SELECT * FROM upload_sessions WHERE status = $1 LIMIT 1', ['processing']);
    if (rows.length > 0) {
      res.json({ sessionId: rows[0].session_id });
    } else {
      res.json({ sessionId: null });
    }
  } catch (error) {
    console.error('Error checking active upload sessions:', error);
    res.status(500).json({ error: 'Database error', details: error.message });
  }
});

// Serve a status homepage
app.get('/', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>PanicSense API</title>
      <style>
        body {
          font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
          line-height: 1.6;
        }
        h1 { color: #333; }
        .card {
          background: #f5f5f5;
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .endpoint {
          background: #e9e9e9;
          padding: 8px 12px;
          border-radius: 4px;
          font-family: monospace;
          display: inline-block;
          margin: 4px 0;
        }
      </style>
    </head>
    <body>
      <h1>PanicSense API Server</h1>
      <div class="card">
        <h2>Status: Online ✅</h2>
        <p>The PanicSense API server is running successfully.</p>
      </div>
      
      <div class="card">
        <h2>Available Endpoints</h2>
        <div><span class="endpoint">/api/disaster-events</span> - Get all disaster events</div>
        <div><span class="endpoint">/api/sentiment-posts</span> - Get sentiment analysis posts</div>
        <div><span class="endpoint">/api/analyzed-files</span> - Get analyzed files</div>
        <div><span class="endpoint">/api/active-upload-session</span> - Check for active upload sessions</div>
      </div>
      
      <div class="card">
        <p>PanicSense Disaster Intelligence Platform © ${new Date().getFullYear()}</p>
      </div>
    </body>
    </html>
  `);
});

// Start server
async function startServer() {
  // Verify database connection
  await testDatabaseConnection();
  
  // Start server
  server.listen(port, '0.0.0.0', () => {
    console.log(`✅ Server running on port ${port}`);
    console.log(`✅ Server URL: http://0.0.0.0:${port}`);
    console.log(`✅ NODE_ENV: ${process.env.NODE_ENV || 'development'}`);
  });
}

// Error handling
process.on('unhandledRejection', (err) => {
  console.error('Unhandled Rejection:', err);
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});

// Start the server
startServer();