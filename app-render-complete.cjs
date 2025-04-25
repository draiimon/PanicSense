/**
 * Complete Render.com compatible server for PanicSense
 * Includes Python service integration for full functionality
 */

const express = require('express');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { spawn } = require('child_process');
const { Pool } = require('pg');
const multer = require('multer');
const WebSocket = require('ws');
const session = require('express-session');
const os = require('os');

// Use development mode as requested
process.env.NODE_ENV = 'development';
const PORT = process.env.PORT || 10000;

// Display startup information
console.log('========================================');
console.log(`🚀 STARTING PANICSENSE FULL SERVER ON PORT ${PORT}`);
console.log(`📅 Time: ${new Date().toISOString()}`);
console.log(`🌍 NODE_ENV: ${process.env.NODE_ENV}`);
console.log(`💻 System: ${os.platform()} ${os.release()}`);
console.log('========================================');

// Create database connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

// Test database connection
async function testDatabase() {
  try {
    const client = await pool.connect();
    const result = await client.query('SELECT NOW() as now');
    console.log(`✅ Database connected. Server time: ${result.rows[0].now}`);
    client.release();
    return true;
  } catch (error) {
    console.error('❌ Database connection error:', error.message);
    return false;
  }
}

// Create Express application
const app = express();
const server = http.createServer(app);

// Configure WebSocket server
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      console.log('WebSocket message received:', data.type);
    } catch (e) {
      console.error('Invalid WebSocket message:', e);
    }
  });
  
  ws.on('close', () => {
    console.log('WebSocket client disconnected');
  });
});

// Configure file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniquePrefix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniquePrefix + '-' + file.originalname);
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB limit
});

// Configure middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(session({
  secret: process.env.SESSION_SECRET || 'panicsense-development-secret',
  resave: false,
  saveUninitialized: true,
  cookie: { maxAge: 24 * 60 * 60 * 1000 } // 1 day
}));

// Enable CORS for development
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  next();
});

// Python service
let pythonProcess = null;

function startPythonService() {
  try {
    // Check if Python exists
    const pythonPath = process.env.PYTHON_PATH || 'python';
    
    // Use the daemon script which doesn't require command-line arguments
    const scriptPath = path.join(__dirname, 'python', 'daemon.py');
    const fallbackPath = path.join(__dirname, 'dist', 'python', 'daemon.py');
    
    let foundScript = false;
    let actualPath = '';
    
    if (fs.existsSync(scriptPath)) {
      console.log(`✅ Found Python daemon script at ${scriptPath}`);
      actualPath = scriptPath;
      foundScript = true;
    } else if (fs.existsSync(fallbackPath)) {
      console.log(`✅ Found Python daemon script at fallback path: ${fallbackPath}`);
      actualPath = fallbackPath;
      foundScript = true;
    } else {
      console.log(`❌ Python daemon script not found, checking alternate locations...`);
      
      // Try to find the script in various locations
      const possiblePaths = [
        path.join(__dirname, 'python'),
        path.join(__dirname, 'dist', 'python'),
        path.join(__dirname, '..', 'python'),
        path.join('/opt/render/project/src/python')
      ];
      
      for (const dirPath of possiblePaths) {
        try {
          if (fs.existsSync(dirPath)) {
            const files = fs.readdirSync(dirPath);
            console.log(`Files in ${dirPath}:`, files);
            
            if (files.includes('daemon.py')) {
              actualPath = path.join(dirPath, 'daemon.py');
              foundScript = true;
              console.log(`✅ Found Python daemon script at: ${actualPath}`);
              break;
            } else if (files.includes('process.py')) {
              actualPath = path.join(dirPath, 'process.py');
              foundScript = true;
              console.log(`✅ Found Python process script at: ${actualPath}`);
              break;
            }
          }
        } catch (error) {
          console.error(`Error checking ${dirPath}:`, error.message);
        }
      }
    }
    
    if (foundScript) {
      console.log(`Starting Python service using: ${pythonPath} ${actualPath}`);
      pythonProcess = spawn(pythonPath, [actualPath]);
      
      pythonProcess.stdout.on('data', (data) => {
        console.log(`Python: ${data.toString().trim()}`);
      });
      
      pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data.toString().trim()}`);
      });
      
      pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        // Auto-restart the Python process if it exits
        if (code !== 0) {
          console.log('Restarting Python service...');
          setTimeout(() => {
            startPythonService();
          }, 5000); // Wait 5 seconds before restarting
        }
        pythonProcess = null;
      });
      
      return true;
    } else {
      console.error(`❌ No Python scripts found!`);
      return false;
    }
  } catch (error) {
    console.error('Failed to start Python service:', error);
    return false;
  }
}

// API Routes
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    version: '1.0.0',
    time: new Date().toISOString(),
    env: process.env.NODE_ENV,
    pythonActive: pythonProcess !== null,
    databaseConnected: pool ? true : false
  });
});

// File upload endpoint
app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    console.log(`File uploaded: ${req.file.originalname}`);
    
    // Store in database
    const client = await pool.connect();
    const result = await client.query(
      'INSERT INTO uploaded_files (original_name, stored_name, file_path) VALUES ($1, $2, $3) RETURNING id',
      [req.file.originalname, req.file.filename, req.file.path]
    );
    client.release();
    
    res.json({
      success: true,
      fileId: result.rows[0].id,
      message: 'File uploaded successfully'
    });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: 'Upload failed', message: error.message });
  }
});

// Get analyzed files
app.get('/api/analyzed-files', async (req, res) => {
  try {
    const client = await pool.connect();
    const result = await client.query('SELECT * FROM analyzed_files ORDER BY created_at DESC');
    client.release();
    res.json(result.rows);
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({ error: 'Failed to retrieve analyzed files' });
  }
});

// Get sentiment posts
app.get('/api/sentiment-posts', async (req, res) => {
  try {
    const client = await pool.connect();
    const result = await client.query('SELECT * FROM sentiment_posts ORDER BY timestamp DESC LIMIT 100');
    client.release();
    res.json(result.rows);
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({ error: 'Failed to retrieve sentiment posts' });
  }
});

// Get disaster events
app.get('/api/disaster-events', async (req, res) => {
  try {
    const client = await pool.connect();
    const result = await client.query('SELECT * FROM disaster_events ORDER BY created_at DESC');
    client.release();
    res.json(result.rows);
  } catch (error) {
    console.error('Database error:', error);
    res.status(500).json({ error: 'Failed to retrieve disaster events' });
  }
});

// Serve static files from the built frontend
const distPath = path.join(__dirname, 'dist', 'public');
if (fs.existsSync(path.join(distPath, 'index.html'))) {
  console.log(`✅ Found frontend files in: ${distPath}`);
  app.use(express.static(distPath));
} else {
  console.error('❌ WARNING: No frontend files found!');
}

// Handle all other routes for SPA
app.get('*', (req, res) => {
  // Skip API routes that weren't matched
  if (req.path.startsWith('/api/')) {
    return res.status(404).json({ error: 'API endpoint not found' });
  }
  
  // For all other routes, serve the SPA's index.html
  const indexPath = path.join(distPath, 'index.html');
  if (fs.existsSync(indexPath)) {
    res.sendFile(indexPath);
  } else {
    res.status(404).send('Frontend not found. Build may be incomplete.');
  }
});

// Function to broadcast updates
function broadcastUpdate(data) {
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(data));
    }
  });
}

// Initialize the server
async function initServer() {
  // Test database connection
  const dbConnected = await testDatabase();
  
  // Start Python service if possible
  const pythonStarted = startPythonService();
  console.log(`Python service status: ${pythonStarted ? 'started' : 'failed to start'}`);
  
  // Start the server
  server.listen(PORT, '0.0.0.0', () => {
    console.log('========================================');
    console.log(`🚀 SERVER RUNNING AT: http://0.0.0.0:${PORT}`);
    console.log(`📅 Server ready at: ${new Date().toISOString()}`);
    console.log(`📊 Database connected: ${dbConnected}`);
    console.log(`🐍 Python service: ${pythonStarted ? 'active' : 'inactive'}`);
    console.log('========================================');
  });
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('Received SIGINT signal, shutting down gracefully...');
  
  // Terminate Python process if running
  if (pythonProcess) {
    pythonProcess.kill();
  }
  
  server.close(() => {
    pool.end();
    console.log('Server closed, database connections terminated');
    process.exit(0);
  });
});

process.on('SIGTERM', () => {
  console.log('Received SIGTERM signal, shutting down gracefully...');
  
  // Terminate Python process if running
  if (pythonProcess) {
    pythonProcess.kill();
  }
  
  server.close(() => {
    pool.end();
    console.log('Server closed, database connections terminated');
    process.exit(0);
  });
});

// Start the application
initServer().catch(error => {
  console.error('Failed to initialize server:', error);
  process.exit(1);
});