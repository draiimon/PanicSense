import { Express, Request, Response } from 'express';
import { RealNewsService } from '../utils/real-news-feed';
// Import AI disaster detector from TypeScript implementation
import aiDisasterDetector from '../utils/ai-disaster-detector';
// Import the disaster news filter for Groq API validation
import { isDisasterNews } from '../utils/disaster-news-filter';

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
      
      // Filter out non-disaster news using Groq API
      const filteredNewsItems = [];
      
      // Process each news item to check if it's a disaster
      for (const newsItem of newsItems) {
        try {
          const validation = await isDisasterNews(newsItem.title, newsItem.content);
          
          if (validation.isDisaster) {
            // Add additional metadata from the validation
            filteredNewsItems.push({
              ...newsItem,
              validatedAsDisaster: true,
              disasterConfidence: validation.confidence,
              disasterType: validation.disasterType,
              validationDetails: validation.details
            });
          }
        } catch (validationError) {
          console.warn(`Error validating news item: ${newsItem.title}`, validationError);
          // Skip items that failed validation
        }
      }
      
      // Ensure we have at least some news items even if validation fails
      const itemsToAnalyze = filteredNewsItems.length > 0 
        ? filteredNewsItems.slice(0, limit) 
        : newsItems.slice(0, limit);
      
      // Use AI to analyze the news items (limited to avoid rate limits)
      const analyzedItems = await aiDisasterDetector.analyzeBatch(itemsToAnalyze, limit);
      
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
      
      // Filter out non-disaster news for the top items using Groq API
      try {
        const topLimit = Math.min(10, sorted.length);
        const filteredItems = [];
        
        // Validate each news item to check if it's a disaster
        for (const item of sorted.slice(0, topLimit)) {
          try {
            const validation = await isDisasterNews(item.title, item.content);
            
            if (validation.isDisaster) {
              // Add additional metadata for the validated items
              filteredItems.push({
                ...item,
                validatedAsDisaster: true,
                disasterConfidence: validation.confidence,
                disasterType: validation.disasterType,
                validationDetails: validation.details
              });
            }
          } catch (validationError) {
            console.warn(`Error validating combined feed item: ${item.title}`, validationError);
            // Skip items that failed validation
          }
        }
        
        // If we have validated disaster news, use those for the top items
        if (filteredItems.length > 0) {
          // Get the non-top items that weren't filtered
          const nonTopItems = sorted.slice(topLimit);
          
          // Construct the new sorted array with filtered disaster news first
          sorted.splice(0, sorted.length, ...filteredItems, ...nonTopItems);
        } else {
          // If validation didn't yield any disaster news, fall back to AI analysis
          const limit = Math.min(5, sorted.length);
          const topItems = sorted.slice(0, limit);
          const analyzedTopItems = await aiDisasterDetector.analyzeBatch(topItems, limit);
          
          // Replace the original top items with the analyzed ones
          // Use type assertion to handle compatibility
          sorted.splice(0, limit, ...(analyzedTopItems as any[]));
        }
      } catch (analysisError) {
        console.warn('Non-fatal error during news validation/analysis:', analysisError);
        // Continue with unfiltered news if validation/analysis fails
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
      
      // Use Groq API to validate if news is disaster-related
      let validatedAlerts = [];
      
      try {
        // First pass: filter using traditional keyword matching to reduce API calls
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
        
        // Second pass: validate with Groq API
        for (const item of keywordFilteredItems.slice(0, 15)) { // Limit to 15 items to avoid too many API calls
          try {
            const validation = await isDisasterNews(item.title, item.content);
            
            if (validation.isDisaster && validation.confidence > 0.6) {
              validatedAlerts.push({
                ...item,
                validatedAsDisaster: true,
                disasterConfidence: validation.confidence,
                disasterType: validation.disasterType,
                validationDetails: validation.details
              });
            }
          } catch (validationError) {
            console.warn(`Error validating news item for disaster alerts: ${item.title}`, validationError);
          }
        }
        
        // If we don't have enough validated alerts, fall back to AI analysis
        if (validatedAlerts.length < 5) {
          // Process a smaller set with AI to avoid rate limits
          const itemsToAnalyze = keywordFilteredItems
            .filter(item => !validatedAlerts.some(alert => alert.id === item.id))
            .slice(0, 10);
          
          // Analyze with AI
          const analyzedItems = await aiDisasterDetector.analyzeBatch(itemsToAnalyze, 10);
          
          // Filter for high confidence disaster items
          const aiAlerts = analyzedItems.filter(item => 
            item.analysis && 
            item.analysis.is_disaster_related && 
            item.analysis.confidence > 0.6 &&
            item.analysis.severity >= 3
          );
          
          // Combine validated alerts with AI alerts
          validatedAlerts = [...validatedAlerts, ...aiAlerts];
        }
        
      } catch (validationError) {
        console.warn('Error using Groq API for disaster validation, falling back to keyword matching:', validationError);
        
        // Fallback to basic keyword matching
        validatedAlerts = newsItems.filter(item => {
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
      const sorted = [...validatedAlerts].sort((a, b) => {
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