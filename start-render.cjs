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
const { simpleDbFix } = require('./server/db-simple-fix.cjs');
const multer = require('multer');
const pg = require('pg');
const pgSession = require('connect-pg-simple')(session);

// DEVELOPMENT MODE with detailed error logging
process.env.NODE_ENV = 'development';
process.env.DEBUG = 'express:*,drizzle:*,postgres:*,neon:*,pg:*';
const PORT = process.env.PORT || 10000;

// Log detailed environment information
console.log('========================================');
console.log(`🚀 [RENDER] STARTING PANICSENSE IN DEVELOPMENT MODE`);
console.log(`📅 Time: ${new Date().toISOString()}`);
console.log(`🔌 PORT: ${PORT}`);
console.log(`🌍 NODE_ENV: ${process.env.NODE_ENV}`);

// Initialize Neon database connection
console.log('========================================');
console.log('🗄️ DATABASE CONFIGURATION:');
console.log(`DB Connection: ${process.env.DATABASE_URL ? 'CONFIGURED' : 'MISSING'}`);
console.log(`Neon Connection: ${process.env.NEON_DATABASE_URL ? 'CONFIGURED' : 'MISSING'}`);

// Set up the database configuration for Neon PG
const neonConfig = {
  webSocketConstructor: ws
};

// Prioritize Neon database URL if available, fall back to regular DATABASE_URL
let databaseUrl = process.env.NEON_DATABASE_URL || process.env.DATABASE_URL;

if (!databaseUrl) {
  console.error('❌ NO DATABASE URL FOUND! Please set DATABASE_URL or NEON_DATABASE_URL');
  process.exit(1);
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
    console.error('❌ Cannot proceed without database connection');
    process.exit(1);
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
  
  // Setup session with PostgreSQL
  app.use(
    session({
      store: new pgSession({
        pool,
        tableName: 'session', // Use a custom session table name
        createTableIfMissing: true
      }),
      secret: process.env.SESSION_SECRET || 'render-secure-panicsense-cat',
      resave: false,
      saveUninitialized: true,
      cookie: { 
        secure: false,
        maxAge: 30 * 24 * 60 * 60 * 1000 // 30 days
      }
    })
  );
  
  // Import server routes
  console.log('🔄 Loading API routes...');
  try {
    // Use the direct CommonJS routes file rather than compiled ESM
    console.log("🔍 Using pure CommonJS routes for maximum compatibility");
    
    // First try the direct routes.cjs file
    const cjsRoutesPath = path.join(__dirname, 'server', 'routes.cjs');
    
    // If routes.cjs doesn't exist in server directory, try dist directory
    const distCjsRoutesPath = path.join(__dirname, 'dist', 'routes.cjs');
    
    // Then try the compiled routes
    const compiledRoutesPath = path.join(__dirname, 'dist', 'routes.js');
    
    console.log("🔍 Looking for routes files at:");
    console.log(` - ${cjsRoutesPath}`);
    console.log(` - ${distCjsRoutesPath}`);
    console.log(` - ${compiledRoutesPath}`);
    
    let serverRoutes;
    
    // Try each option in order of preference
    if (fs.existsSync(cjsRoutesPath)) {
      console.log(`✅ Found CommonJS routes at: ${cjsRoutesPath}`);
      serverRoutes = require(cjsRoutesPath);
    }
    else if (fs.existsSync(distCjsRoutesPath)) {
      console.log(`✅ Found CommonJS routes in dist at: ${distCjsRoutesPath}`);
      serverRoutes = require(distCjsRoutesPath);
    }
    else if (fs.existsSync(compiledRoutesPath)) {
      try {
        console.log(`✅ Found compiled routes at: ${compiledRoutesPath}`);
        serverRoutes = require(compiledRoutesPath);
      } catch (error) {
        console.error(`❌ Error loading compiled routes:`, error.message);
        // Default minimal API as fallback
        serverRoutes = getFallbackRoutes();
      }
    } 
    else {
      console.error('❌ No routes file found! Using minimal API implementation');
      serverRoutes = getFallbackRoutes();
    }
    
    // Check if registerRoutes function exists
    if (typeof serverRoutes.registerRoutes === 'function') {
      await serverRoutes.registerRoutes(app);
      console.log('✅ API routes registered successfully');
    } else {
      console.error('❌ registerRoutes function not found in server routes');
    }
  } catch (error) {
    console.error('❌ Error loading API routes:', error);
  }
  
  // Serve static files
  const distPath = path.join(__dirname, 'dist', 'public');
  if (fs.existsSync(path.join(distPath, 'index.html'))) {
    console.log(`✅ Found frontend files in: ${distPath}`);
    app.use(express.static(distPath));
  } else {
    console.error('❌ WARNING: No frontend files found!');
  }
  
  // Define fallback API routes
  app.get('/api/health', (req, res) => {
    res.json({ 
      status: 'ok', 
      time: new Date().toISOString(),
      env: process.env.NODE_ENV,
      databaseConnected: dbConnected,
      databaseType: databaseUrl.split(':')[0]
    });
  });
  
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
    console.log(`🚀 SERVER RUNNING IN DEVELOPMENT MODE`);
    console.log(`📡 Server listening at: http://0.0.0.0:${PORT}`);
    console.log(`📅 Server ready at: ${new Date().toISOString()}`);
    console.log('========================================');
  });
}

// Helper function to get fallback routes
function getFallbackRoutes() {
  return {
    registerRoutes: async (app) => {
      console.log('📝 Registering minimal API routes...');
      app.get('/api/health', (req, res) => {
        res.json({ status: 'ok', mode: 'fallback' });
      });
      return app;
    }
  };
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