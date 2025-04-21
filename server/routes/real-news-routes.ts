import { Express, Request, Response } from 'express';
import { RealNewsService } from '../utils/real-news-feed';
// Import AI disaster detector from TypeScript implementation
import aiDisasterDetector from '../utils/ai-disaster-detector';

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
  
  // Get AI-analyzed disaster news (enhanced with AI classification)
  app.get('/api/ai-disaster-news', async (req: Request, res: Response) => {
    try {
      // Get the latest news
      const newsItems = await realNewsService.getLatestNews();
      
      // Get the limit parameter (default to 5)
      const limit = req.query.limit ? parseInt(req.query.limit as string, 10) : 5;
      
      // Use AI to analyze the news items (limited to avoid rate limits)
      const analyzedItems = await aiDisasterDetector.analyzeBatch(newsItems, limit);
      
      // Sort by timestamp (newest first)
      const sorted = [...analyzedItems].sort((a, b) => {
        const dateA = new Date(a.timestamp || (a as any).publishedAt || "").getTime();
        const dateB = new Date(b.timestamp || (b as any).publishedAt || "").getTime();
        return dateB - dateA;
      });
      
      res.json(sorted);
    } catch (error) {
      console.error(`Error getting AI-analyzed disaster news:`, error);
      res.status(500).json({ 
        error: 'Could not retrieve AI-analyzed disaster news', 
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
        const dateA = new Date(a.timestamp || (a as any).publishedAt || "").getTime();
        const dateB = new Date(b.timestamp || (b as any).publishedAt || "").getTime();
        return dateB - dateA;
      });
      
      // Try to analyze the top 5 news items with AI for disaster detection
      try {
        const limit = Math.min(5, sorted.length);
        const topItems = sorted.slice(0, limit);
        const analyzedTopItems = await aiDisasterDetector.analyzeBatch(topItems, limit);
        
        // Replace the original top items with the analyzed ones
        // Use type assertion to handle compatibility
        sorted.splice(0, limit, ...(analyzedTopItems as any[]));
      } catch (analysisError) {
        console.warn('Non-fatal error during news analysis:', analysisError);
        // Continue with unanalyzed news if AI analysis fails
      }
      
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
      
      // Use AI for more accurate disaster alerts detection
      let alerts = [];
      
      try {
        // First, filter using traditional keyword matching to reduce API calls
        const keywordFilteredItems = newsItems.filter(item => {
          // Check for emergency keywords in title or content
          return [
            'emergency', 'alert', 'warning', 'evacuate', 'evacuación',
            'danger', 'severe', 'critical', 'imminent', 'immediate',
            'typhoon', 'earthquake', 'flood', 'fire', 'landslide', 
            'volcano', 'tsunami', 'drought'
          ].some(keyword => 
            item.title.toLowerCase().includes(keyword) ||
            item.content.toLowerCase().includes(keyword)
          );
        });
        
        // Process a smaller set with AI to avoid rate limits
        const itemsToAnalyze = keywordFilteredItems.slice(0, 10);
        
        // Analyze with AI
        const analyzedItems = await aiDisasterDetector.analyzeBatch(itemsToAnalyze, 10);
        
        // Filter for high confidence disaster items
        alerts = analyzedItems.filter(item => 
          item.analysis.is_disaster_related && 
          item.analysis.confidence > 0.6 &&
          item.analysis.severity >= 3
        );
        
        // If we don't have enough alerts, include some keyword matches
        if (alerts.length < 5) {
          const remainingCount = 5 - alerts.length;
          const analyzedIds = new Set(alerts.map(item => item.id));
          
          // Add more items from keyword matched ones that weren't analyzed
          const additionalAlerts = keywordFilteredItems
            .filter(item => !analyzedIds.has(item.id))
            .slice(0, remainingCount);
            
          alerts = [...alerts, ...additionalAlerts];
        }
        
      } catch (aiError) {
        console.warn('Error using AI for disaster detection, falling back to keyword matching:', aiError);
        
        // Fallback to basic keyword matching
        alerts = newsItems.filter(item => {
          // Check for emergency keywords in title or content
          return [
            'emergency', 'alert', 'warning', 'evacuate', 'evacuación',
            'danger', 'severe', 'critical', 'imminent', 'immediate',
            'typhoon', 'earthquake', 'flood', 'fire', 'landslide', 
            'volcano', 'tsunami', 'drought'
          ].some(keyword => 
            item.title.toLowerCase().includes(keyword) ||
            item.content.toLowerCase().includes(keyword)
          );
        });
      }
      
      // Sort by timestamp (newest first)
      const sorted = [...alerts].sort((a, b) => {
        const dateA = new Date(a.timestamp || (a as any).publishedAt || "").getTime();
        const dateB = new Date(b.timestamp || (b as any).publishedAt || "").getTime();
        return dateB - dateA;
      });
      
      res.json(sorted);
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