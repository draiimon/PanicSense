/**
 * Real News Feed Routes for PanicSense
 * Implements real-time feed functionality from authentic Philippine news sources
 */

import { Request, Response } from 'express';
import { Express } from 'express';
import { WebSocket } from 'ws';
import { storage } from '../storage';
import {
  startRealNewsFeed,
  stopRealNewsFeed,
  manuallyFetchNews,
  getLatestPosts,
  addWebSocketClient
} from '../utils/real-news-feed';

/**
 * Registers real-time news feed routes
 */
export function registerRealNewsRoutes(app: Express, wss: any) {
  // Get the latest disaster-related news posts for the real-time feed (default 20)
  app.get('/api/real-news-feed', async (req: Request, res: Response) => {
    try {
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 20;
      const posts = await getLatestPosts(limit);
      res.json(posts);
    } catch (error) {
      console.error('Error fetching real-time news feed:', error);
      res.status(500).json({ error: 'Failed to fetch real-time news feed' });
    }
  });

  // Start the real news feed service
  app.post('/api/real-news-feed/start', async (req: Request, res: Response) => {
    try {
      startRealNewsFeed();
      res.json({ success: true, message: 'Real news feed service started' });
    } catch (error) {
      console.error('Error starting real news feed service:', error);
      res.status(500).json({ error: 'Failed to start real news feed service' });
    }
  });

  // Stop the real news feed service
  app.post('/api/real-news-feed/stop', async (req: Request, res: Response) => {
    try {
      stopRealNewsFeed();
      res.json({ success: true, message: 'Real news feed service stopped' });
    } catch (error) {
      console.error('Error stopping real news feed service:', error);
      res.status(500).json({ error: 'Failed to stop real news feed service' });
    }
  });

  // Manually fetch news right now
  app.post('/api/real-news-feed/fetch-now', async (req: Request, res: Response) => {
    try {
      const posts = await manuallyFetchNews();
      res.json({ 
        success: true, 
        message: 'Real news fetched successfully', 
        count: posts.length,
        posts: posts.slice(0, 5) // Just return the 5 most recent for preview
      });
    } catch (error) {
      console.error('Error fetching news manually:', error);
      res.status(500).json({ error: 'Failed to fetch news manually' });
    }
  });

  // Register WebSocket connection handler for real-time news updates
  wss.on('connection', (ws: WebSocket, req: Request) => {
    // Check if this connection is for the real-time news feed
    const url = req.url || '';
    if (url.includes('/real-news-feed-socket')) {
      addWebSocketClient(ws);
      
      // Send a welcome message
      ws.send(JSON.stringify({
        type: 'connection_established',
        message: 'Connected to real-time natural disaster news feed',
        timestamp: new Date().toISOString()
      }));
    }
  });
}