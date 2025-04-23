/**
 * Simple CommonJS version of routes.ts 
 * This is a compatibility layer for production deployment
 */

const express = require('express');
const { createServer } = require('http');
const { WebSocketServer } = require('ws');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { nanoid } = require('nanoid');

// Configure multer for file uploads
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024, 
  },
  fileFilter: (req, file, cb) => {
    if (file.originalname.toLowerCase().endsWith('.csv')) {
      cb(null, true);
    } else {
      cb(new Error('Only CSV files are allowed'));
    }
  }
});

// Track connected WebSocket clients
const connectedClients = new Set();

// Broadcast function
function broadcastUpdate(data) {
  // Add timestamp to all messages
  data.timestamp = Date.now();
  
  if (data.type === 'progress') {
    try {
      // Enhanced progress object
      const enhancedProgress = {
        type: 'progress',
        timestamp: Date.now(),
        progress: {
          processed: data.progress?.processed || 0,
          total: data.progress?.total || 100,
          stage: data.progress?.stage || 'Processing...',
          timestamp: Date.now()
        }
      };

      // Send to all connected clients
      const message = JSON.stringify(enhancedProgress);
      connectedClients.forEach(client => {
        if (client.readyState === WebSocketServer.OPEN) {
          try {
            client.send(message);
          } catch (error) {
            console.error('Failed to send WebSocket message:', error);
          }
        }
      });
    } catch (error) {
      console.error('Error processing progress update:', error);
    }
  }
}

async function registerRoutes(app) {
  // Add health check endpoint for monitoring
  app.get('/api/health', (req, res) => {
    res.status(200).json({ 
      status: 'healthy',
      timestamp: new Date().toISOString(),
      environment: process.env.NODE_ENV || 'development'
    });
  });
  
  // Serve static files from attached_assets
  app.use('/assets', express.static(path.join(process.cwd(), 'attached_assets')));

  // Create HTTP server
  const httpServer = createServer(app);

  // Create WebSocket server
  const wss = new WebSocketServer({ 
    server: httpServer,
    path: '/ws'  
  });
  
  // WebSocket connection handler
  wss.on('connection', (ws) => {
    console.log('New WebSocket client connected');
    connectedClients.add(ws);

    // Handle client disconnection
    ws.on('close', () => {
      console.log('WebSocket client disconnected');
      connectedClients.delete(ws);
    });

    // Handle client messages
    ws.on('message', (message) => {
      try {
        const data = JSON.parse(message.toString());
        console.log('Received message:', data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    });
  });

  // Basic error handler
  app.use((err, _req, res, _next) => {
    console.error(err.stack);
    res.status(500).json({
      success: false,
      message: 'An internal server error occurred',
      error: process.env.NODE_ENV === 'production' ? undefined : err.message,
    });
  });

  return httpServer;
}

module.exports = { registerRoutes, broadcastUpdate };