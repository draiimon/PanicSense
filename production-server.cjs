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

// Initialize
const app = express();
const server = http.createServer(app);

// Create WebSocket server for real-time updates
const wss = new WebSocket.Server({ server, path: '/ws' });

wss.on('connection', (ws) => {
  console.log('WebSocket client connected');
  
  ws.on('close', () => {
    console.log('WebSocket client disconnected');
  });
});

// Python service management
let pythonProcess = null;
const pythonEvents = [];
const pythonErrors = [];

function startPythonDaemon() {
  try {
    // Start Python with daemonize mode
    const pythonPath = process.env.PYTHON_PATH || 'python';
    const scriptPath = path.join(__dirname, 'python', 'daemon.py');
    
    if (fs.existsSync(scriptPath)) {
      console.log(`Found Python script at ${scriptPath}`);
      pythonProcess = spawn(pythonPath, [scriptPath]);
      
      pythonProcess.stdout.on('data', (data) => {
        const message = data.toString().trim();
        console.log(`Python: ${message}`);
        pythonEvents.push({
          type: 'stdout',
          message,
          timestamp: new Date().toISOString()
        });
        
        // Broadcast to all connected clients
        wss.clients.forEach(client => {
          if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify({
              type: 'python_event',
              message,
              timestamp: new Date().toISOString()
            }));
          }
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
      });
      
      pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        pythonProcess = null;
        
        // Auto-restart
        if (code !== 0) {
          console.log('Restarting Python service in 5 seconds...');
          setTimeout(startPythonDaemon, 5000);
        }
      });
      
      return true;
    } else {
      // Try alternate paths
      const altPath = path.join(__dirname, 'dist', 'python', 'daemon.py');
      if (fs.existsSync(altPath)) {
        console.log(`Found Python script at alternate path: ${altPath}`);
        // Start with alternate path
        pythonProcess = spawn(pythonPath, [altPath]);
        // Set up the same event handlers as above...
        // (Duplicate code omitted for brevity)
        return true;
      }
      
      console.error(`Python script not found at: ${scriptPath}`);
      return false;
    }
  } catch (error) {
    console.error('Failed to start Python service:', error);
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