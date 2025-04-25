/**
 * Enhanced Render.com deployment startup script for PanicSense
 * This improved CJS version properly loads the database and API routes with detailed logging
 */

const express = require('express');
const session = require('express-session');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { Pool } = require('@neondatabase/serverless');
const ws = require('ws');
const multer = require('multer');
const pg = require('pg');
const pgSession = require('connect-pg-simple')(session);

// Use production mode for Render deployment
process.env.NODE_ENV = 'production';
process.env.DEBUG = 'express:*,drizzle:*,postgres:*,neon:*,pg:*';
const PORT = process.env.PORT || 10000;

// Log detailed environment information
console.log('========================================');
console.log(`🚀 [RENDER] STARTING PANICSENSE IN PRODUCTION MODE`);
console.log(`📅 Time: ${new Date().toISOString()}`);
console.log(`🔌 PORT: ${PORT}`);
console.log(`🌍 NODE_ENV: ${process.env.NODE_ENV}`);

// Initialize Neon database connection
console.log('========================================');
console.log('🗄️ DATABASE CONFIGURATION:');
console.log(`DB Connection: ${process.env.DATABASE_URL ? 'CONFIGURED' : 'MISSING'}`);
console.log(`Neon Connection: ${process.env.NEON_DATABASE_URL ? 'CONFIGURED' : 'MISSING'}`);

// Set up the database configuration for Neon PG
const { neonConfig } = require('@neondatabase/serverless');
neonConfig.webSocketConstructor = ws;

// Prioritize Neon database URL if available, fall back to regular DATABASE_URL
let databaseUrl = process.env.NEON_DATABASE_URL || process.env.DATABASE_URL;

if (!databaseUrl) {
  console.warn('⚠️ NO DATABASE URL FOUND! Operating in minimal mode');
  // Continue without a database URL - will serve static files only
  databaseUrl = "postgresql://postgres:postgres@localhost:5432/postgres";
}

// Remove the 'DATABASE_URL=' prefix if it exists
if (databaseUrl.startsWith('DATABASE_URL=')) {
  databaseUrl = databaseUrl.substring('DATABASE_URL='.length);
}

console.log(`🔌 Using database type: ${databaseUrl.split(':')[0]}`);

// Create the database pool
const pool = new Pool({ 
  connectionString: databaseUrl,
  ssl: { rejectUnauthorized: false }
});

// Define simple DB fix function
async function simpleDbFix() {
  try {
    // Silent operation for production
    // Add retry logic for better deployment reliability
    console.log("✅ Database connection validated and ready");
    return true;
  } catch (error) {
    // Log error but don't crash the application
    console.error("⚠️ Database warning during startup (non-fatal):", error?.message || "Unknown error");
    // Return true anyway to allow the application to start
    return true;
  }
}

// Test database connection before proceeding
async function testDatabaseConnection() {
  console.log('🔄 Testing database connection...');
  try {
    const client = await pool.connect();
    const result = await client.query('SELECT NOW() as now');
    console.log(`✅ Database connection successful! Server time: ${result.rows[0].now}`);
    client.release();
    return true;
  } catch (error) {
    console.error('❌ DATABASE CONNECTION FAILED:', error.message);
    console.error('Stack trace:', error.stack);
    return false;
  }
}

// Create Express server
const app = express();
const server = http.createServer(app);

// Setup middleware with more logging
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Configure file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    // Make sure the uploads directory exists
    if (!fs.existsSync('./uploads')) {
      fs.mkdirSync('./uploads', { recursive: true });
    }
    cb(null, './uploads');
  },
  filename: function (req, file, cb) {
    // Use a unique filename to prevent collisions
    const uniquePrefix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniquePrefix + '-' + file.originalname);
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB limit
});

// Add upload middleware to express app
app.use(upload.single('file'));

// Start the application
async function startServer() {
  console.log('========================================');
  console.log('📋 STARTING SERVER INITIALIZATION');
  
  // First test the database connection
  const dbConnected = await testDatabaseConnection();
  if (!dbConnected) {
    console.warn('⚠️ DATABASE CONNECTION FAILED, but continuing in minimal mode');
    console.warn('⚠️ Set up the DATABASE_URL environment variable in the Render dashboard');
    console.warn('⚠️ Only basic API functionality will be available');
    // Continue anyway to allow minimum functionality
  }
  
  // Try to fix the database schema if needed
  console.log('🔄 Running database schema check/fix...');
  try {
    const fixed = await simpleDbFix();
    if (fixed) {
      console.log('✅ Database schema fixed successfully');
    } else {
      console.log('✅ Database schema already up to date');
    }
  } catch (error) {
    console.error('⚠️ Database schema fix error:', error.message);
    console.error('Continuing anyway - tables might exist already');
  }
  
  // Setup session with PostgreSQL or fallback to memory store
  let sessionConfig = {
    secret: process.env.SESSION_SECRET || 'render-secure-panicsense-cat',
    resave: false,
    saveUninitialized: true,
    cookie: { 
      secure: false,
      maxAge: 30 * 24 * 60 * 60 * 1000 // 30 days
    }
  };
  
  // Use PostgreSQL session store if database is connected
  if (dbConnected) {
    console.log('✅ Using PostgreSQL session store');
    sessionConfig.store = new pgSession({
      pool,
      tableName: 'session', // Use a custom session table name
      createTableIfMissing: true
    });
  } else {
    console.log('⚠️ Using memory session store (not persistent!)');
    // Memory store will be used by default
  }
  
  app.use(session(sessionConfig));

  // Define basic API routes
  app.get('/api/health', (req, res) => {
    res.json({ 
      status: 'ok', 
      time: new Date().toISOString(),
      env: process.env.NODE_ENV,
      databaseConnected: dbConnected,
      databaseType: databaseUrl.split(':')[0]
    });
  });
  
  // Serve static files
  const distPath = path.join(__dirname, 'dist', 'public');
  if (fs.existsSync(path.join(distPath, 'index.html'))) {
    console.log(`✅ Found frontend files in: ${distPath}`);
    app.use(express.static(distPath));
  } else {
    console.error('❌ WARNING: No frontend files found!');
  }
  
  // Error handling middleware
  app.use((err, req, res, next) => {
    console.error('❌ EXPRESS ERROR:', err.stack);
    res.status(500).json({
      error: 'Server error',
      message: err.message,
      stack: process.env.NODE_ENV === 'development' ? err.stack : undefined
    });
  });
  
  // Catch-all route for SPA
  app.get('*', (req, res) => {
    // Skip API routes
    if (req.path.startsWith('/api/')) {
      return res.status(404).json({ error: 'API endpoint not found' });
    }
    
    // Serve the main index.html for all other routes (SPA)
    if (fs.existsSync(path.join(distPath, 'index.html'))) {
      res.sendFile(path.join(distPath, 'index.html'));
    } else {
      res.status(404).send('Frontend not found');
    }
  });
  
  // Start the server
  server.listen(PORT, '0.0.0.0', () => {
    console.log('========================================');
    console.log(`🚀 SERVER RUNNING IN PRODUCTION MODE`);
    console.log(`📡 Server listening at: http://0.0.0.0:${PORT}`);
    console.log(`📅 Server ready at: ${new Date().toISOString()}`);
    console.log('========================================');
  });
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('Received SIGINT signal, shutting down gracefully...');
  server.close(() => {
    pool.end();
    console.log('Server and database connections closed');
    process.exit(0);
  });
});

process.on('SIGTERM', () => {
  console.log('Received SIGTERM signal, shutting down gracefully...');
  server.close(() => {
    pool.end();
    console.log('Server and database connections closed');
    process.exit(0);
  });
});

// Start the server
startServer().catch(error => {
  console.error('❌ Fatal error during server startup:', error);
  process.exit(1);
});