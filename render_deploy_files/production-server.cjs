/**
 * Combined Express + Python server for Render deployment
 * No credit card needed - runs as a single service
 */

const express = require('express');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { spawn } = require('child_process');
const WebSocket = require('ws');

// Configure environment
const PORT = process.env.PORT || 10000;
const NODE_ENV = process.env.NODE_ENV || 'development';
const DEBUG = process.env.DEBUG === 'true';

// Initialize
const app = express();
const server = http.createServer(app);

// Debug logging
function debug(message) {
  if (DEBUG) {
    console.log(`[DEBUG] ${message}`);
  }
}

// Enhanced console logging
console.originalLog = console.log;
console.log = (...args) => {
  const timestamp = new Date().toISOString();
  console.originalLog(`[${timestamp}]`, ...args);
};

console.originalError = console.error;
console.error = (...args) => {
  const timestamp = new Date().toISOString();
  console.originalError(`[ERROR ${timestamp}]`, ...args);
};

// Create WebSocket server for real-time updates
const wss = new WebSocket.Server({ server, path: '/ws' });

wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  
  // Send initial data upon connection
  ws.send(JSON.stringify({
    type: 'connection_established',
    timestamp: new Date().toISOString(),
    pythonActive: pythonProcess !== null,
    recentEvents: pythonEvents.slice(-5),
    recentErrors: pythonErrors.slice(-5)
  }));
  
  ws.on('close', () => {
    console.log('WebSocket client disconnected');
  });
  
  // Handle messages from client
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      
      // Handle ping
      if (data.type === 'ping') {
        ws.send(JSON.stringify({
          type: 'pong',
          timestamp: new Date().toISOString(),
          pythonActive: pythonProcess !== null
        }));
      }
      
      // Handle explicit requests for python status
      if (data.type === 'get_python_status') {
        ws.send(JSON.stringify({
          type: 'python_status',
          active: pythonProcess !== null,
          events: pythonEvents.slice(-10),
          errors: pythonErrors.slice(-10),
          timestamp: new Date().toISOString()
        }));
      }
    } catch (error) {
      console.error('Error processing WebSocket message:', error);
    }
  });
});

// Send updates to all connected clients
function broadcastUpdate(data) {
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({
        ...data,
        timestamp: new Date().toISOString()
      }));
    }
  });
}

// Python service management
let pythonProcess = null;
const pythonEvents = [];
const pythonErrors = [];
let pythonStartAttempts = 0;

function findPythonExecutable() {
  // Try different Python executable names
  const possiblePythons = ['python', 'python3', 'python3.11', 'python3.10', 'python3.9'];
  
  for (const pythonName of possiblePythons) {
    try {
      const result = require('child_process').spawnSync(pythonName, ['--version']);
      if (result.status === 0) {
        console.log(`Found Python executable: ${pythonName}`);
        return pythonName;
      }
    } catch (error) {
      debug(`Python executable ${pythonName} not available`);
    }
  }
  
  // If we get here, default to 'python'
  console.log('No Python executable found, defaulting to "python"');
  return 'python';
}

function startPythonDaemon() {
  pythonStartAttempts++;
  console.log(`Starting Python daemon (attempt ${pythonStartAttempts})...`);
  
  try {
    // List all directories to help debug
    debug('Current directory structure:');
    if (DEBUG) {
      try {
        const files = fs.readdirSync('.');
        debug(`Files in current directory: ${files.join(', ')}`);
        
        if (fs.existsSync('./python')) {
          const pythonFiles = fs.readdirSync('./python');
          debug(`Files in python directory: ${pythonFiles.join(', ')}`);
        }
      } catch (error) {
        debug(`Error listing directories: ${error.message}`);
      }
    }
    
    // Find Python executable
    const pythonPath = process.env.PYTHON_PATH || findPythonExecutable();
    console.log(`Using Python executable: ${pythonPath}`);
    
    // Find the daemon script
    const possiblePaths = [
      path.join(__dirname, 'python', 'daemon.py'),
      path.join(__dirname, 'dist', 'python', 'daemon.py'),
      path.join('python', 'daemon.py'),
      path.join('dist', 'python', 'daemon.py'),
      path.resolve(__dirname, 'python', 'daemon.py')
    ];
    
    let scriptPath = null;
    for (const p of possiblePaths) {
      if (fs.existsSync(p)) {
        scriptPath = p;
        console.log(`Found Python script at: ${scriptPath}`);
        break;
      }
    }
    
    if (!scriptPath) {
      throw new Error(`Python script not found at any of the expected locations: ${possiblePaths.join(', ')}`);
    }
    
    // Make the script executable just in case
    try {
      fs.chmodSync(scriptPath, '755');
      console.log(`Made ${scriptPath} executable`);
    } catch (error) {
      debug(`Unable to chmod script: ${error.message}`);
    }
    
    // Start the Python process
    console.log(`Spawning Python process: ${pythonPath} ${scriptPath}`);
    pythonProcess = spawn(pythonPath, [scriptPath], {
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1'  // Important for real-time logging
      }
    });
    
    pythonProcess.stdout.on('data', (data) => {
      const message = data.toString().trim();
      console.log(`Python: ${message}`);
      pythonEvents.push({
        type: 'stdout',
        message,
        timestamp: new Date().toISOString()
      });
      
      // Keep events list from growing too large
      if (pythonEvents.length > 1000) {
        pythonEvents.shift();
      }
      
      // Broadcast to all connected clients
      broadcastUpdate({
        type: 'python_event',
        message
      });
    });
    
    pythonProcess.stderr.on('data', (data) => {
      const message = data.toString().trim();
      console.error(`Python Error: ${message}`);
      pythonErrors.push({
        type: 'stderr',
        message,
        timestamp: new Date().toISOString()
      });
      
      // Keep errors list from growing too large
      if (pythonErrors.length > 1000) {
        pythonErrors.shift();
      }
      
      // Broadcast error to all connected clients
      broadcastUpdate({
        type: 'python_error',
        message
      });
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
      pythonProcess = null;
      
      // Broadcast the exit to all connected clients
      broadcastUpdate({
        type: 'python_exit',
        code
      });
      
      // Auto-restart with exponential backoff
      if (pythonStartAttempts < 10) {
        const delay = Math.min(30000, Math.pow(2, pythonStartAttempts) * 1000);
        console.log(`Restarting Python service in ${delay/1000} seconds...`);
        setTimeout(startPythonDaemon, delay);
      } else {
        console.error('Too many Python restart attempts, giving up');
        pythonErrors.push({
          type: 'restart_failure',
          message: 'Too many restart attempts, gave up restarting Python',
          timestamp: new Date().toISOString()
        });
      }
    });
    
    pythonProcess.on('error', (error) => {
      console.error(`Python process spawn error: ${error.message}`);
      pythonErrors.push({
        type: 'spawn_error',
        message: error.message,
        timestamp: new Date().toISOString()
      });
      
      pythonProcess = null;
    });
    
    console.log('Python daemon started successfully');
    return true;
  } catch (error) {
    console.error('Failed to start Python service:', error);
    pythonErrors.push({
      type: 'start_error',
      message: error.message,
      timestamp: new Date().toISOString()
    });
    return false;
  }
}

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

// API endpoints
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    version: '1.0.0',
    time: new Date().toISOString(),
    env: NODE_ENV,
    pythonActive: pythonProcess !== null,
    pythonEvents: pythonEvents.slice(-5), // Last 5 events
    pythonErrors: pythonErrors.slice(-5)  // Last 5 errors
  });
});

// Text analysis API
app.post('/api/analyze', (req, res) => {
  const { text } = req.body;
  
  if (!text) {
    return res.status(400).json({ error: 'Text is required' });
  }
  
  if (!pythonProcess) {
    return res.status(503).json({ error: 'Python service is not running' });
  }
  
  // In a real implementation, you'd send the text to Python for processing
  // For now, just return sample data
  res.json({
    text,
    results: {
      sentiment: 'panic',
      confidence: 0.87,
      disasterType: 'flood',
      location: 'Manila',
      language: 'Taglish'
    },
    timestamp: new Date().toISOString()
  });
});

// Disaster events
app.get('/api/disaster-events', (req, res) => {
  // Return stored events or sample data
  res.json(
    // Using sample data here, in production this would come from your database
    [
      {
        id: 1,
        name: 'Typhoon Yolanda Alert',
        description: 'Category 5 typhoon approaching Visayas region',
        location: 'Eastern Visayas',
        severity: 'Severe',
        timestamp: new Date().toISOString()
      },
      {
        id: 2, 
        name: 'Mayon Volcano Eruption',
        description: 'Ongoing volcanic activity with ash fall in surrounding areas',
        location: 'Albay, Bicol Region',
        severity: 'High',
        timestamp: new Date().toISOString()
      },
      {
        id: 3,
        name: 'Flood in Manila',
        description: 'Heavy rainfall causing flooding in metro Manila areas',
        location: 'Metro Manila',
        severity: 'Moderate',
        timestamp: new Date().toISOString()
      }
    ]
  );
});

// Get sentiment posts
app.get('/api/sentiment-posts', (req, res) => {
  res.json([
    {
      id: 1,
      text: "Grabe ang baha dito sa Marikina! Tulong po! #FloodPH",
      timestamp: new Date().toISOString(),
      source: "Twitter",
      language: "Taglish",
      sentiment: "Panic",
      confidence: 0.89,
      explanation: "Expresses panic and urgent call for help",
      disasterType: "Flood",
      location: "Marikina"
    },
    {
      id: 2,
      text: "Alert Level 4 na ang Taal Volcano. Evacuation ongoing sa Batangas.",
      timestamp: new Date().toISOString(),
      source: "Facebook",
      language: "Taglish",
      sentiment: "Alert",
      confidence: 0.92,
      explanation: "Reporting alert level and evacuation",
      disasterType: "Volcanic Activity",
      location: "Batangas"
    },
    {
      id: 3,
      text: "6.2 magnitude earthquake just hit Davao region. Everyone stay safe!",
      timestamp: new Date().toISOString(),
      source: "Twitter",
      language: "English",
      sentiment: "Alert",
      confidence: 0.85,
      explanation: "Reporting earthquake and expressing concern",
      disasterType: "Earthquake",
      location: "Davao"
    }
  ]);
});

// Analyzed files
app.get('/api/analyzed-files', (req, res) => {
  res.json([
    {
      id: 1,
      originalName: "disaster_tweets.csv",
      storedName: "disaster_tweets_processed.csv",
      recordCount: 1251,
      created_at: new Date().toISOString(),
      accuracy: 0.89,
      f1Score: 0.87
    },
    {
      id: 2,
      originalName: "emergency_reports.csv",
      storedName: "emergency_reports_processed.csv",
      recordCount: 532,
      created_at: new Date().toISOString(),
      accuracy: 0.91,
      f1Score: 0.90
    }
  ]);
});

// Python logs API
app.get('/api/python-logs', (req, res) => {
  res.json({
    events: pythonEvents,
    errors: pythonErrors,
    active: pythonProcess !== null
  });
});

// Get upload status
app.get('/api/active-upload-session', (req, res) => {
  res.json({ sessionId: null });
});

// Serve static files
const distPath = path.join(__dirname, 'dist', 'public');
if (fs.existsSync(path.join(distPath, 'index.html'))) {
  console.log(`✅ Found frontend files in: ${distPath}`);
  app.use(express.static(distPath));
} else {
  console.error(`❌ Frontend files not found at: ${distPath}`);
  // Check alternative locations
  const altLocations = [
    path.join(__dirname, 'public'),
    path.join(__dirname, 'client', 'dist'),
    path.join(__dirname, 'client', 'public')
  ];
  
  let foundFrontend = false;
  for (const location of altLocations) {
    if (fs.existsSync(path.join(location, 'index.html'))) {
      console.log(`✅ Found frontend files in alternate location: ${location}`);
      app.use(express.static(location));
      foundFrontend = true;
      break;
    }
  }
  
  if (!foundFrontend) {
    console.error('❌ Frontend files not found in any standard location');
  }
}

// Catch-all route for SPA
app.get('*', (req, res) => {
  if (req.path.startsWith('/api/')) {
    return res.status(404).json({ error: 'API endpoint not found' });
  }
  
  // Try to find an index.html file in various locations
  const locations = [
    path.join(distPath, 'index.html'),
    path.join(__dirname, 'public', 'index.html'),
    path.join(__dirname, 'client', 'dist', 'index.html'),
    path.join(__dirname, 'client', 'public', 'index.html')
  ];
  
  for (const location of locations) {
    if (fs.existsSync(location)) {
      return res.sendFile(location);
    }
  }
  
  // If no index.html found, serve a basic status page
  res.status(200).send(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>PanicSense API Server</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #d9534f; }
        .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .method { display: inline-block; padding: 3px 6px; border-radius: 3px; background: #007bff; color: white; }
        .status { padding: 5px 10px; border-radius: 4px; display: inline-block; }
        .status.ok { background-color: #28a745; color: white; }
        .status.error { background-color: #dc3545; color: white; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
      </style>
    </head>
    <body>
      <h1>PanicSense API Server</h1>
      <p>Server is running in ${NODE_ENV} mode. Frontend files not found, but API endpoints are available:</p>
      
      <div class="endpoint">
        <span class="method">GET</span> <a href="/api/health">/api/health</a> - Check API health
      </div>
      
      <div class="endpoint">
        <span class="method">GET</span> <a href="/api/disaster-events">/api/disaster-events</a> - Get active disasters
      </div>
      
      <div class="endpoint">
        <span class="method">GET</span> <a href="/api/sentiment-posts">/api/sentiment-posts</a> - Get sentiment data
      </div>
      
      <div class="endpoint">
        <span class="method">GET</span> <a href="/api/analyzed-files">/api/analyzed-files</a> - Get analyzed files
      </div>
      
      <div class="endpoint">
        <span class="method">GET</span> <a href="/api/python-logs">/api/python-logs</a> - Python service logs
      </div>
      
      <h2>Python Service Status</h2>
      <div class="status ${pythonProcess ? 'ok' : 'error'}">
        ${pythonProcess ? 'RUNNING' : 'STOPPED'}
      </div>
      
      <h2>Recent Python Events</h2>
      <pre>${JSON.stringify(pythonEvents.slice(-5), null, 2)}</pre>
      
      <h2>Recent Python Errors</h2>
      <pre>${JSON.stringify(pythonErrors.slice(-5), null, 2)}</pre>
      
      <p>Server started at: ${new Date().toISOString()}</p>
    </body>
    </html>
  `);
});

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 PanicSense Server running on http://0.0.0.0:${PORT}`);
  console.log(`📅 Started at: ${new Date().toISOString()}`);
  console.log(`🌍 Environment: ${NODE_ENV}`);
  
  // Start Python service
  const pythonStarted = startPythonDaemon();
  console.log(`🐍 Python service: ${pythonStarted ? 'STARTED' : 'FAILED TO START'}`);
});

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('Received SIGINT signal, shutting down gracefully');
  
  if (pythonProcess) {
    pythonProcess.kill();
  }
  
  server.close(() => {
    process.exit(0);
  });
});

process.on('SIGTERM', () => {
  console.log('Received SIGTERM signal, shutting down gracefully');
  
  if (pythonProcess) {
    pythonProcess.kill();
  }
  
  server.close(() => {
    process.exit(0);
  });
});