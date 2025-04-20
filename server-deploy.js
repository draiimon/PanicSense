/**
 * SIMPLE PRODUCTION SERVER FOR RENDER DEPLOYMENT
 * Using ES modules for compatibility with package.json "type": "module"
 */

import express from 'express';
import pkg from 'pg';
const { Pool } = pkg;
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
const publicDir = path.join(__dirname, '../dist/public');

// Check if the directory exists and serve it
if (fs.existsSync(publicDir)) {
  console.log(`Serving static files from ${publicDir}`);
  app.use(express.static(publicDir));
  
  // SPA fallback
  app.get('*', (req, res) => {
    const indexHtml = path.join(publicDir, 'index.html');
    if (fs.existsSync(indexHtml)) {
      res.sendFile(indexHtml);
    } else {
      console.error('Error: index.html not found in', publicDir);
      res.send('Application is running but index.html not found. Check server logs.');
    }
  });
} else {
  console.error(`Error: Static directory not found at ${publicDir}`);
  // List all directories to debug
  console.log('Available directories:');
  try {
    const rootDir = path.join(__dirname, '..');
    const dirs = fs.readdirSync(rootDir);
    console.log(dirs);
  } catch (err) {
    console.error('Error listing directories:', err);
  }
  
  app.get('*', (req, res) => {
    res.send('Application is running but static files directory not found. Check server logs.');
  });
}

// Start server
app.listen(port, '0.0.0.0', () => {
  console.log(`Server running on port ${port}`);
});