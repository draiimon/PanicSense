import { Express, Request, Response } from 'express';
import { RealNewsService } from '../utils/real-news-feed';
import socialMediaMonitor from '../utils/social-media-monitor';

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

  // Get social media updates
  app.get('/api/social-media/posts', async (req: Request, res: Response) => {
    try {
      const posts = await socialMediaMonitor.getLatestPosts();
      res.json(posts);
    } catch (error) {
      console.error(`Error getting social media posts:`, error);
      res.status(500).json({ 
        error: 'Could not retrieve social media updates', 
        message: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
  });

  // Get combined feed (news + social media)
  app.get('/api/combined-feed', async (req: Request, res: Response) => {
    try {
      const [newsItems, socialPosts] = await Promise.all([
        realNewsService.getLatestNews(),
        socialMediaMonitor.getLatestPosts()
      ]);
      
      // Combine the feeds and sort by timestamp (newest first)
      const combined = [...newsItems, ...socialPosts].sort((a, b) => {
        const dateA = 'timestamp' in a ? new Date(a.timestamp).getTime() : new Date(a.date).getTime();
        const dateB = 'timestamp' in b ? new Date(b.timestamp).getTime() : new Date(b.date).getTime();
        return dateB - dateA;
      });
      
      res.json(combined);
    } catch (error) {
      console.error(`Error getting combined feed:`, error);
      res.status(500).json({ 
        error: 'Could not retrieve combined updates', 
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

  console.log('Real news feed routes registered successfully');
  return Promise.resolve();
}