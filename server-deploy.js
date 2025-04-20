/**
 * SIMPLE PRODUCTION SERVER FOR RENDER DEPLOYMENT
 * Using ES modules for compatibility with package.json "type": "module"
 */

import express from 'express';
import { Pool } from 'pg';
import http from 'http';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

// ES module equivalent for __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Create Express app
const app = express();
const port = process.env.PORT || 10000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Connect to database
let pool;
if (process.env.DATABASE_URL) {
  pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: { rejectUnauthorized: false }
  });
  console.log('Database connection initialized');
}

// API routes
app.get('/api/health', async (req, res) => {
  try {
    if (pool) {
      const client = await pool.connect();
      await client.query('SELECT NOW()');
      client.release();
      res.json({ status: 'ok', database: 'connected' });
    } else {
      res.json({ status: 'ok', database: 'not configured' });
    }
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Serve static files
const clientDir = path.join(__dirname, 'client/dist');
if (fs.existsSync(clientDir)) {
  console.log(`Serving static files from ${clientDir}`);
  app.use(express.static(clientDir));
  
  // SPA fallback
  app.get('*', (req, res) => {
    res.sendFile(path.join(clientDir, 'index.html'));
  });
} else {
  console.warn(`Static directory ${clientDir} not found`);
  app.get('*', (req, res) => {
    res.send('Application is running but frontend is not built correctly');
  });
}

// Start server
app.listen(port, '0.0.0.0', () => {
  console.log(`Server running on port ${port}`);
});