/**
 * Real-time News and Social Media Routes
 * Handles endpoints for news sources and social media monitoring
 */

import { Express, Request, Response } from 'express';
import * as WebSocket from 'ws';
import { 
  startRealNewsFeed, 
  stopRealNewsFeed, 
  getLatestPosts, 
  manuallyFetchNews 
} from '../utils/real-news-feed';

import {
  startSocialMediaMonitor,
  stopSocialMediaMonitor,
  getLatestSocialMediaPosts,
  manuallyFetchTweets
} from '../utils/social-media-monitor';

import { log } from '../vite';

export async function registerRealNewsRoutes(app: Express, wss?: WebSocket.Server): Promise<void> {
  try {
    // Start real-time news feed and social media monitor on server start
    startRealNewsFeed();
    startSocialMediaMonitor();
    
    // Get latest posts from real news sources
    app.get('/api/real-news/posts', async (req: Request, res: Response) => {
      try {
        const limit = req.query.limit ? parseInt(req.query.limit as string) : 20;
        const posts = await getLatestPosts(limit);
        res.json(posts);
      } catch (error) {
        log(`Error getting real news posts: ${error}`, 'real-news-routes');
        res.status(500).json({ error: 'Failed to fetch real news posts' });
      }
    });
    
    // Manually trigger news fetch
    app.post('/api/real-news/fetch', async (req: Request, res: Response) => {
      try {
        const posts = await manuallyFetchNews();
        res.json({ 
          success: true, 
          message: `Successfully fetched ${posts.length} news items`, 
          posts 
        });
      } catch (error) {
        log(`Error manually fetching news: ${error}`, 'real-news-routes');
        res.status(500).json({ error: 'Failed to fetch news' });
      }
    });
    
    // Get latest posts from social media
    app.get('/api/social-media/posts', async (req: Request, res: Response) => {
      try {
        const limit = req.query.limit ? parseInt(req.query.limit as string) : 10;
        const posts = await getLatestSocialMediaPosts(limit);
        res.json(posts);
      } catch (error) {
        log(`Error getting social media posts: ${error}`, 'real-news-routes');
        res.status(500).json({ error: 'Failed to fetch social media posts' });
      }
    });
    
    // Manually trigger tweets fetch
    app.post('/api/social-media/fetch', async (req: Request, res: Response) => {
      try {
        const posts = await manuallyFetchTweets();
        res.json({ 
          success: true, 
          message: `Successfully fetched ${posts.length} social media posts`, 
          posts 
        });
      } catch (error) {
        log(`Error manually fetching tweets: ${error}`, 'real-news-routes');
        res.status(500).json({ error: 'Failed to fetch tweets' });
      }
    });
    
    // Get combined feed from both news and social media
    app.get('/api/combined-feed', async (req: Request, res: Response) => {
      try {
        const limit = req.query.limit ? parseInt(req.query.limit as string) : 20;
        const newsLimit = Math.ceil(limit * 0.6); // 60% news
        const socialLimit = Math.ceil(limit * 0.4); // 40% social
        
        const [newsPosts, socialPosts] = await Promise.all([
          getLatestPosts(newsLimit),
          getLatestSocialMediaPosts(socialLimit)
        ]);
        
        // Combine and sort by timestamp
        const combinedFeed = [...newsPosts, ...socialPosts].sort((a, b) => {
          return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
        });
        
        // Limit the total
        const limitedFeed = combinedFeed.slice(0, limit);
        
        res.json(limitedFeed);
      } catch (error) {
        log(`Error getting combined feed: ${error}`, 'real-news-routes');
        res.status(500).json({ error: 'Failed to fetch combined feed' });
      }
    });
    
    // Control route to restart both feeds
    app.post('/api/restart-feeds', async (req: Request, res: Response) => {
      try {
        // Stop both feeds
        stopRealNewsFeed();
        stopSocialMediaMonitor();
        
        // Small delay to ensure proper shutdown
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Restart both feeds
        startRealNewsFeed();
        startSocialMediaMonitor();
        
        res.json({ 
          success: true, 
          message: 'Successfully restarted real-time news feed and social media monitor'
        });
      } catch (error) {
        log(`Error restarting feeds: ${error}`, 'real-news-routes');
        res.status(500).json({ error: 'Failed to restart feeds' });
      }
    });
    
    // Statistics for monitoring
    app.get('/api/feed-stats', async (req: Request, res: Response) => {
      try {
        const [newsPosts, socialPosts] = await Promise.all([
          getLatestPosts(100), // Get more to calculate meaningful stats
          getLatestSocialMediaPosts(100)
        ]);
        
        // Calculate statistics
        const stats = {
          newsPostsCount: newsPosts.length,
          socialPostsCount: socialPosts.length,
          lastNewsPostTime: newsPosts.length > 0 ? newsPosts[0].timestamp : null,
          lastSocialPostTime: socialPosts.length > 0 ? socialPosts[0].timestamp : null,
          disasterTypeBreakdown: calculateDisasterBreakdown([...newsPosts, ...socialPosts]),
          locationBreakdown: calculateLocationBreakdown([...newsPosts, ...socialPosts]),
          sourceBreakdown: calculateSourceBreakdown([...newsPosts, ...socialPosts])
        };
        
        res.json(stats);
      } catch (error) {
        log(`Error getting feed stats: ${error}`, 'real-news-routes');
        res.status(500).json({ error: 'Failed to fetch feed statistics' });
      }
    });
    
    log('Real news feed routes registered successfully', 'real-news-routes');
    return Promise.resolve();
  } catch (error) {
    log(`Error registering real news routes: ${error}`, 'real-news-routes');
    return Promise.reject(error);
  }
}

// Helper function to calculate disaster type breakdown
function calculateDisasterBreakdown(posts: any[]): Record<string, number> {
  const breakdown: Record<string, number> = {};
  
  posts.forEach(post => {
    const disasterType = post.disasterType || 'UNKNOWN';
    breakdown[disasterType] = (breakdown[disasterType] || 0) + 1;
  });
  
  return breakdown;
}

// Helper function to calculate location breakdown
function calculateLocationBreakdown(posts: any[]): Record<string, number> {
  const breakdown: Record<string, number> = {};
  
  posts.forEach(post => {
    const location = post.location || 'UNKNOWN';
    breakdown[location] = (breakdown[location] || 0) + 1;
  });
  
  return breakdown;
}

// Helper function to calculate source breakdown
function calculateSourceBreakdown(posts: any[]): Record<string, number> {
  const breakdown: Record<string, number> = {};
  
  posts.forEach(post => {
    const source = post.source || 'UNKNOWN';
    
    // Group similar sources (e.g., Twitter accounts)
    let normalizedSource = source;
    if (source.includes('Twitter/X')) {
      normalizedSource = 'Twitter/X';
    }
    
    breakdown[normalizedSource] = (breakdown[normalizedSource] || 0) + 1;
  });
  
  return breakdown;
}