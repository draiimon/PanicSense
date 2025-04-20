/**
 * Real-time feeds API routes for PanicSense
 * Implements auto-generating posts and real-time feed functionality
 */

import { Request, Response } from 'express';
import { Express } from 'express';
import { WebSocket } from 'ws';
import { storage } from '../storage';
import {
  startAutoPostGenerator,
  stopAutoPostGenerator,
  manuallyGeneratePosts,
  getLatestPosts,
  addWebSocketClient
} from '../utils/auto-post-generator';

/**
 * Registers real-time feed routes
 */
export function registerRealTimeFeedRoutes(app: Express, wss: any) {
  // Get the latest posts for the real-time feed (default 20)
  app.get('/api/real-time-feed', async (req: Request, res: Response) => {
    try {
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 20;
      const posts = await getLatestPosts(limit);
      res.json(posts);
    } catch (error) {
      console.error('Error fetching real-time feed:', error);
      res.status(500).json({ error: 'Failed to fetch real-time feed' });
    }
  });

  // Start the auto-post generator
  app.post('/api/real-time-feed/start', async (req: Request, res: Response) => {
    try {
      startAutoPostGenerator();
      res.json({ success: true, message: 'Auto-post generator started' });
    } catch (error) {
      console.error('Error starting auto-post generator:', error);
      res.status(500).json({ error: 'Failed to start auto-post generator' });
    }
  });

  // Stop the auto-post generator
  app.post('/api/real-time-feed/stop', async (req: Request, res: Response) => {
    try {
      stopAutoPostGenerator();
      res.json({ success: true, message: 'Auto-post generator stopped' });
    } catch (error) {
      console.error('Error stopping auto-post generator:', error);
      res.status(500).json({ error: 'Failed to stop auto-post generator' });
    }
  });

  // Manually generate posts
  app.post('/api/real-time-feed/generate', async (req: Request, res: Response) => {
    try {
      const count = req.body.count ? parseInt(req.body.count) : 3;
      const posts = await manuallyGeneratePosts(count);
      res.json({ success: true, posts, count: posts.length });
    } catch (error) {
      console.error('Error generating posts manually:', error);
      res.status(500).json({ error: 'Failed to generate posts manually' });
    }
  });

  // Listen for WebSocket connections specifically for the real-time feed
  wss.on('connection', (ws: WebSocket) => {
    const url = (ws as any).url || '/';
    
    // Check if this connection is for the real-time feed
    if (url.includes('/ws/real-time-feed')) {
      // Register this client for real-time feed updates
      addWebSocketClient(ws);
      
      // Send initial data
      getLatestPosts(20).then(posts => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: 'initial_feed',
            data: posts,
            timestamp: new Date().toISOString()
          }));
        }
      }).catch(error => {
        console.error('Error sending initial feed data:', error);
      });
    }
  });
}