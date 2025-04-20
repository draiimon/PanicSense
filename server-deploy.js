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
// Try multiple possible static file locations
const possibleDirs = [
  path.join(__dirname, 'public'),             // dist/public when built
  path.join(__dirname, '../public'),          // /public
  path.join(__dirname, '../dist/public'),     // /dist/public 
  path.join(__dirname, 'client/dist'),        // original location
  path.join(__dirname, '../client/dist')      // /client/dist
];

let staticDir = null;

// Find the first directory that exists
for (const dir of possibleDirs) {
  if (fs.existsSync(dir)) {
    staticDir = dir;
    console.log(`Found static files in: ${dir}`);
    break;
  }
}

if (staticDir) {
  console.log(`Serving static files from ${staticDir}`);
  app.use(express.static(staticDir));
  
  // SPA fallback
  app.get('*', (req, res) => {
    if (fs.existsSync(path.join(staticDir, 'index.html'))) {
      res.sendFile(path.join(staticDir, 'index.html'));
    } else {
      res.send('Application is running but index.html not found');
    }
  });
} else {
  console.warn(`No static directory found. Tried: ${possibleDirs.join(', ')}`);
  app.get('*', (req, res) => {
    res.send('Application is running but frontend static files were not found');
  });
}

// Start server
app.listen(port, '0.0.0.0', () => {
  console.log(`Server running on port ${port}`);
});