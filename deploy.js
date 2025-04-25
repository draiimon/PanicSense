/**
 * DIRECT DEPLOYMENT SCRIPT FOR RENDER
 * This is a standalone server that works reliably on Render
 */

const express = require('express');
const path = require('path');
const fs = require('fs');
const http = require('http');

const app = express();
const server = http.createServer(app);
const PORT = process.env.PORT || 10000;

// Configure middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Enable CORS for development
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  next();
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    version: '1.0.0',
    time: new Date().toISOString(),
    env: process.env.NODE_ENV || 'development'
  });
});

// All necessary API endpoints
app.get('/api/disaster-events', (req, res) => {
  // Return some sample data
  res.json([
    {
      id: 1,
      name: 'Typhoon Yolanda Alert',
      description: 'Category 5 typhoon approaching Visayas region',
      location: 'Eastern Visayas',
      severity: 'Severe',
      timestamp: new Date().toISOString(),
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    },
    {
      id: 2,
      name: 'Mayon Volcano Eruption',
      description: 'Ongoing volcanic activity with ash fall in surrounding areas',
      location: 'Albay, Bicol Region',
      severity: 'High',
      timestamp: new Date().toISOString(),
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    },
    {
      id: 3,
      name: 'Flood in Manila',
      description: 'Heavy rainfall causing flooding in metro Manila areas',
      location: 'Metro Manila',
      severity: 'Moderate',
      timestamp: new Date().toISOString(),
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }
  ]);
});

// Sentiment posts endpoint
app.get('/api/sentiment-posts', (req, res) => {
  // Return some sample data
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

// Analyzed files endpoint
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

// Serve static files
const distPath = path.join(__dirname, 'dist', 'public');
if (fs.existsSync(path.join(distPath, 'index.html'))) {
  console.log(`✅ Found frontend files in: ${distPath}`);
  app.use(express.static(distPath));
} else {
  console.error(`❌ Frontend files not found at ${distPath}`);
  
  // Check other possible locations
  const altDistPath = path.join(__dirname, 'public');
  if (fs.existsSync(path.join(altDistPath, 'index.html'))) {
    console.log(`✅ Found frontend files in alternate location: ${altDistPath}`);
    app.use(express.static(altDistPath));
  } else {
    console.error(`❌ Frontend files not found at alternate location: ${altDistPath}`);
  }
}

// Catch-all route
app.get('*', (req, res) => {
  if (req.path.startsWith('/api/')) {
    return res.status(404).json({ error: 'API endpoint not found' });
  }
  
  // Try to serve index.html
  const indexPath = path.join(distPath, 'index.html');
  if (fs.existsSync(indexPath)) {
    res.sendFile(indexPath);
  } else {
    const altIndexPath = path.join(__dirname, 'public', 'index.html');
    if (fs.existsSync(altIndexPath)) {
      res.sendFile(altIndexPath);
    } else {
      res.status(200).send(`
        <!DOCTYPE html>
        <html>
        <head>
          <title>PanicSense Emergency API</title>
          <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #d9534f; }
            .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .method { display: inline-block; padding: 3px 6px; border-radius: 3px; background: #007bff; color: white; }
            a { color: #007bff; }
          </style>
        </head>
        <body>
          <h1>PanicSense API</h1>
          <p>This is the PanicSense Emergency Response API. The frontend is not available, but the API endpoints are working:</p>
          
          <div class="endpoint">
            <span class="method">GET</span> <a href="/api/health">/api/health</a> - Check API health
          </div>
          
          <div class="endpoint">
            <span class="method">GET</span> <a href="/api/disaster-events">/api/disaster-events</a> - Get active disasters
          </div>
          
          <div class="endpoint">
            <span class="method">GET</span> <a href="/api/sentiment-posts">/api/sentiment-posts</a> - Get sentiment analysis data
          </div>
          
          <div class="endpoint">
            <span class="method">GET</span> <a href="/api/analyzed-files">/api/analyzed-files</a> - Get analyzed file reports
          </div>
          
          <p>Deployed version: ${new Date().toISOString()}</p>
        </body>
        </html>
      `);
    }
  }
});

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Server running on http://0.0.0.0:${PORT}`);
  console.log(`📅 Started at: ${new Date().toISOString()}`);
});

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('Received SIGINT signal, shutting down gracefully');
  server.close(() => {
    process.exit(0);
  });
});

process.on('SIGTERM', () => {
  console.log('Received SIGTERM signal, shutting down gracefully');
  server.close(() => {
    process.exit(0);
  });
});