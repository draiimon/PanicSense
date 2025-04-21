import { Express, Request, Response } from 'express';
import { RealNewsService } from '../utils/real-news-feed';

// Initialize our monitoring services
const realNewsService = new RealNewsService();

// Setup routes for real-time news feeds
export async function registerRealNewsRoutes(app: Express): Promise<void> {
  // Get news updates from official sources
  app.get('/api/real-news/posts', async (req: Request, res: Response) => {
    try {
      const newsItems = await realNewsService.getLatestNews();
      res.json(newsItems);
    } catch (error) {
      console.error(`Error getting real news posts:`, error);
      res.status(500).json({ 
        error: 'Could not retrieve news updates', 
        message: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
  });

  // Get combined feed (news only, renamed for backward compatibility)
  app.get('/api/combined-feed', async (req: Request, res: Response) => {
    try {
      const newsItems = await realNewsService.getLatestNews();
      
      // Sort by timestamp (newest first)
      const sorted = [...newsItems].sort((a, b) => {
        const dateA = new Date(a.timestamp || a.publishedAt || "").getTime();
        const dateB = new Date(b.timestamp || b.publishedAt || "").getTime();
        return dateB - dateA;
      });
      
      res.json(sorted);
    } catch (error) {
      console.error(`Error getting news feed:`, error);
      res.status(500).json({ 
        error: 'Could not retrieve news updates', 
        message: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
  });

  // Get disaster alerts (high priority news only)
  app.get('/api/disaster-alerts', async (req: Request, res: Response) => {
    try {
      const newsItems = await realNewsService.getLatestNews();
      
      // Filter for high priority disaster news
      const alerts = newsItems.filter(item => {
        // Check for emergency keywords in title or content
        const isEmergency = [
          'emergency', 'alert', 'warning', 'evacuate', 'evacuación',
          'danger', 'severe', 'critical', 'imminent', 'immediate'
        ].some(keyword => 
          item.title.toLowerCase().includes(keyword) ||
          item.content.toLowerCase().includes(keyword)
        );
        
        return isEmergency;
      });
      
      res.json(alerts);
    } catch (error) {
      console.error(`Error getting disaster alerts:`, error);
      res.status(500).json({ 
        error: 'Could not retrieve disaster alerts', 
        message: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
  });

  // Enhanced logging for development
  console.log('=== REAL NEWS SERVICE INITIALIZATION ===');
  console.log('✅ Real news feed routes registered successfully');
  
  // Schedule periodic news fetching
  const FETCH_INTERVAL = 10 * 60 * 1000; // 10 minutes
  
  // Initial fetch on startup
  setTimeout(async () => {
    try {
      console.log('🔄 Performing initial news fetch...');
      const news = await realNewsService.getLatestNews();
      console.log(`📰 Retrieved ${news.length} news items from sources`);
      
      // Log sources breakdown
      const sourceCounts: Record<string, number> = {};
      news.forEach(item => {
        const source = item.source || 'unknown';
        sourceCounts[source] = (sourceCounts[source] || 0) + 1;
      });
      
      Object.entries(sourceCounts).forEach(([source, count]) => {
        console.log(`📊 Source: ${source}, Items: ${count}`);
      });
    } catch (error) {
      console.error('❌ Error during initial news fetch:', error);
    }
  }, 5000); // 5 second delay after server start
  
  // Schedule periodic fetches to keep data fresh
  console.log(`⏰ Scheduling news fetches every ${FETCH_INTERVAL/60000} minutes`);
  setInterval(async () => {
    try {
      console.log('🔄 Performing scheduled news fetch...');
      const news = await realNewsService.getLatestNews();
      console.log(`📰 Retrieved ${news.length} news items`);
    } catch (error) {
      console.error('❌ Error during scheduled news fetch:', error);
    }
  }, FETCH_INTERVAL);
  
  return Promise.resolve();
}