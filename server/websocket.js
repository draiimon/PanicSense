
/**
 * RELIABLE WEBSOCKET SERVER
 * Provides a direct WebSocket server for real-time updates
 */

import { WebSocketServer } from 'ws';
import http from 'http';

// Create WebSocket server
export function createWebSocketServer(server) {
  // Create WebSocket server on a separate path to avoid conflicts with Vite
  const wss = new WebSocketServer({ 
    server,
    path: '/ws',
    // Set a very long timeout
    clientTracking: true
  });
  
  console.log('WebSocket server created on path: /ws');
  
  // Keep track of clients
  const clients = new Set();
  
  // Handle connections
  wss.on('connection', (ws) => {
    console.log('WebSocket client connected');
    clients.add(ws);
    
    // Send welcome message
    ws.send(JSON.stringify({
      type: 'connection',
      message: 'Connected to PanicSense real-time server',
      timestamp: new Date().toISOString()
    }));
    
    // Handle messages
    ws.on('message', (message) => {
      try {
        const data = JSON.parse(message);
        console.log('Received message:', data);
        
        // Echo back to confirm receipt
        ws.send(JSON.stringify({
          type: 'echo',
          original: data,
          timestamp: new Date().toISOString()
        }));
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    });
    
    // Handle close
    ws.on('close', () => {
      console.log('WebSocket client disconnected');
      clients.delete(ws);
    });
    
    // Handle errors
    ws.on('error', (error) => {
      console.error('WebSocket error:', error);
      clients.delete(ws);
    });
    
    // Send periodic heartbeat to keep connection alive
    const heartbeatInterval = setInterval(() => {
      if (ws.readyState === ws.OPEN) {
        ws.send(JSON.stringify({
          type: 'heartbeat',
          timestamp: new Date().toISOString()
        }));
      } else {
        clearInterval(heartbeatInterval);
        clients.delete(ws);
      }
    }, 30000); // Every 30 seconds
  });
  
  // Broadcast function to send messages to all clients
  const broadcast = (data) => {
    const message = typeof data === 'string' ? data : JSON.stringify(data);
    
    for (const client of clients) {
      if (client.readyState === client.OPEN) {
        client.send(message);
      }
    }
  };
  
  // Return the WebSocket server and broadcast function
  return { wss, broadcast };
}
