/**
 * PRODUCTION SERVER
 * Simple Express server to serve the built app in production
 */

import express from 'express';
import pg from 'pg';
const { Pool } = pg;
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

// ES module equivalent for __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Create Express app
const app = express();
const port = process.env.PORT || 5000;

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

// API health check route
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
const publicDir = path.join(__dirname, 'dist/public');
if (fs.existsSync(publicDir)) {
  console.log(`Serving static files from ${publicDir}`);
  app.use(express.static(publicDir));
  
  // SPA fallback
  app.get('*', (req, res) => {
    res.sendFile(path.join(publicDir, 'index.html'));
  });
} else {
  console.error(`Static directory not found at ${publicDir}`);
  app.get('*', (req, res) => {
    res.send('Application is running but static files are not built. Please run npm run build first.');
  });
}

// Start server
app.listen(port, '0.0.0.0', () => {
  console.log(`Production server running on port ${port}`);
});