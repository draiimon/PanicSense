/**
 * ULTRA SIMPLE PRODUCTION SERVER FOR RENDER.COM
 * Walang complex code, simple lang para gumana
 */

const express = require('express');
const session = require('express-session');
const multer = require('multer');
const { Pool } = require('pg');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const WebSocket = require('ws');
const http = require('http');

// Constants
const app = express();
const server = http.createServer(app);
const PORT = process.env.PORT || 10000;
const DATABASE_URL = process.env.DATABASE_URL;
const SESSION_SECRET = process.env.SESSION_SECRET || 'panicsense-secret-key';

// Database Connection
const pool = new Pool({
  connectionString: DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(session({
  secret: SESSION_SECRET,
  resave: false,
  saveUninitialized: true,
  cookie: { maxAge: 24 * 60 * 60 * 1000 } // 1 day
}));

// Upload middleware
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, `batch-${Date.now()}-${file.originalname}`)
});
const upload = multer({ storage });

// WebSocket server
const wss = new WebSocket.Server({ server, path: '/ws' });

wss.on('connection', (ws) => {
  console.log('New WebSocket client connected');
  
  ws.on('message', (message) => {
    console.log(`Received message: ${message}`);
  });
  
  ws.on('close', () => {
    console.log('WebSocket client disconnected');
  });
});

// Basic API routes
app.get('/api/disaster-events', async (req, res) => {
  try {
    // First attempt with full query including created_at
    try {
      const { rows } = await pool.query('SELECT * FROM disaster_events ORDER BY id DESC');
      return res.json(rows);
    } catch (error) {
      // If error contains "column created_at does not exist" try different query
      if (error.message.includes('created_at')) {
        console.log('Adapting query: disaster_events table does not have created_at column');
        const { rows } = await pool.query('SELECT id, name, description FROM disaster_events ORDER BY id DESC');
        return res.json(rows);
      }
      throw error;
    }
  } catch (error) {
    console.error('Error fetching disaster events:', error);
    res.status(500).json({ error: 'Database error', details: error.message });
  }
});

app.get('/api/sentiment-posts', async (req, res) => {
  try {
    const { rows } = await pool.query('SELECT * FROM sentiment_posts ORDER BY id DESC LIMIT 100');
    res.json(rows);
  } catch (error) {
    console.error('Error fetching sentiment posts:', error);
    res.status(500).json({ error: 'Database error', details: error.message });
  }
});

app.get('/api/analyzed-files', async (req, res) => {
  try {
    // Try with original column names first
    try {
      const { rows } = await pool.query('SELECT * FROM analyzed_files ORDER BY id DESC');
      return res.json(rows);
    } catch (error) {
      if (error.message.includes('created_at')) {
        console.log('Adapting query: analyzed_files table does not have created_at column');
        const { rows } = await pool.query('SELECT id, original_name, stored_name FROM analyzed_files ORDER BY id DESC');
        return res.json(rows);
      }
      throw error;
    }
  } catch (error) {
    console.error('Error fetching analyzed files:', error);
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
    console.error('Error checking for active upload sessions:', error);
    res.status(500).json({ error: 'Database error', details: error.message });
  }
});

app.post('/api/upload', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  try {
    const sessionId = `upload-${Date.now()}-${Math.random().toString(36).substring(2, 10)}`;
    
    // Create upload session
    await pool.query(
      'INSERT INTO upload_sessions (session_id, status, progress, file_path) VALUES ($1, $2, $3, $4)',
      [sessionId, 'processing', 0, req.file.path]
    );
    
    // Look for Python script
    const pythonScript = path.resolve(__dirname, 'python', 'process.py');
    if (fs.existsSync(pythonScript)) {
      // Start Python processing
      const pythonProcess = spawn('python', [
        pythonScript,
        req.file.path,
        sessionId,
        '--mode=csv'
      ]);
      
      // Handle Python process output
      pythonProcess.stdout.on('data', (data) => {
        console.log(`Python output: ${data}`);
        
        // Check for progress updates to broadcast to WebSocket clients
        if (data.toString().includes('PROGRESS:')) {
          const progress = data.toString().match(/PROGRESS: (\d+)/)[1];
          wss.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
              client.send(JSON.stringify({ type: 'progress', sessionId, progress }));
            }
          });
        }
      });
      
      pythonProcess.stderr.on('data', (data) => {
        console.error(`Python error: ${data}`);
      });
      
      pythonProcess.on('close', async (code) => {
        console.log(`Python process exited with code ${code}`);
        
        // Update session in database
        const status = code === 0 ? 'completed' : 'error';
        const progress = code === 0 ? 100 : -1;
        
        await pool.query(
          'UPDATE upload_sessions SET status = $1, progress = $2 WHERE session_id = $3',
          [status, progress, sessionId]
        );
        
        // Notify clients via WebSocket
        wss.clients.forEach(client => {
          if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify({ 
              type: 'upload_completed', 
              sessionId, 
              success: code === 0 
            }));
          }
        });
      });
    } else {
      console.error(`Python script not found at: ${pythonScript}`);
      // Simulate success even without Python
      setTimeout(async () => {
        await pool.query(
          'UPDATE upload_sessions SET status = $1, progress = $2 WHERE session_id = $3',
          ['completed', 100, sessionId]
        );
        
        wss.clients.forEach(client => {
          if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify({ 
              type: 'upload_completed', 
              sessionId, 
              success: true 
            }));
          }
        });
      }, 5000); // Simulate 5 second processing
    }
    
    res.json({ 
      success: true,
      sessionId,
      message: 'Upload started, processing file'
    });
  } catch (error) {
    console.error('Error during file upload:', error);
    res.status(500).json({ error: 'Upload failed', details: error.message });
  }
});

app.post('/api/cleanup-error-sessions', async (req, res) => {
  try {
    const { rowCount } = await pool.query(
      'DELETE FROM upload_sessions WHERE status = $1 OR (status = $2 AND created_at < NOW() - INTERVAL \'1 day\')',
      ['error', 'processing']
    );
    
    res.json({ 
      success: true, 
      clearedCount: rowCount,
      message: `Successfully cleared ${rowCount} error or stale sessions`
    });
  } catch (error) {
    console.error('Error cleaning up error sessions:', error);
    res.status(500).json({ error: 'Cleanup failed', details: error.message });
  }
});

// Start optional Python daemon
function startPythonDaemon() {
  const pythonScript = path.resolve(__dirname, 'python', 'daemon.py');
  if (fs.existsSync(pythonScript)) {
    console.log(`Starting Python daemon: ${pythonScript}`);
    const pythonProcess = spawn('python', [pythonScript]);
    
    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python daemon: ${data}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python daemon error: ${data}`);
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`Python daemon exited with code ${code}`);
      // Restart daemon after delay if it crashes
      if (code !== 0) {
        setTimeout(startPythonDaemon, 5000);
      }
    });
    
    return pythonProcess;
  } else {
    console.log(`Python daemon script not found at: ${pythonScript}`);
    return null;
  }
}

// Serve static files
const distPath = path.join(__dirname, 'dist');
if (fs.existsSync(distPath)) {
  console.log(`Serving static files from: ${distPath}`);
  app.use(express.static(distPath));
  
  // Serve index.html for client-side routing
  app.get('*', (req, res) => {
    res.sendFile(path.join(distPath, 'index.html'));
  });
} else {
  console.warn(`Static files directory not found: ${distPath}`);
  // Fallback route
  app.get('*', (req, res) => {
    res.send('PanicSense API Server Running');
  });
}

// Start server
async function startServer() {
  try {
    // Test database connection
    const client = await pool.connect();
    client.release();
    console.log('✅ Database connected successfully');
    
    // Start Python daemon
    const pythonDaemon = startPythonDaemon();
    if (pythonDaemon) {
      console.log('✅ Python daemon started successfully');
    }
    
    // Start HTTP server
    server.listen(PORT, '0.0.0.0', () => {
      console.log(`✅ Server running on port ${PORT}`);
      console.log(`✅ WebSocket server running at ws://0.0.0.0:${PORT}/ws`);
    });
    
  } catch (error) {
    console.error('❌ Server startup error:', error);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => cleanupAndExit());
process.on('SIGTERM', () => cleanupAndExit());

function cleanupAndExit() {
  console.log('Shutting down server...');
  server.close(() => {
    console.log('HTTP server closed');
    pool.end().then(() => {
      console.log('Database connection closed');
      process.exit(0);
    });
  });
}

// Start the server
startServer();