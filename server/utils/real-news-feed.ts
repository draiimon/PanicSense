/**
 * Real News Feed Service for PanicSense App
 * Fetches authentic news content from Philippine news sources
 */

import Parser from 'rss-parser';
import { storage } from '../storage';
import { pythonService } from '../python-service';
import { WebSocket } from 'ws';
import { log } from '../vite';

// Connected WebSocket clients that will receive real-time updates
const connectedClients = new Set<WebSocket>();

// Create RSS parser instance
const parser = new Parser({
  headers: {
    'User-Agent': 'PanicSense Disaster Monitoring App/1.0'
  }
});

// List of Philippines news sources with disaster-related content
const NEWS_SOURCES = [
  {
    name: 'GMA News - Nation',
    url: 'https://www.gmanetwork.com/news/rss/news/nation/',
    source: 'GMA News'
  },
  {
    name: 'Manila Bulletin - NDRRMC',
    url: 'https://mb.com.ph/category/news/national/ndrrmc/feed/',
    source: 'Manila Bulletin'
  },
  {
    name: 'Philippine News Agency',
    url: 'https://www.pna.gov.ph/feed/all',
    source: 'PNA'
  },
  {
    name: 'ReliefWeb Philippines',
    url: 'https://reliefweb.int/country/phl/rss.xml',
    source: 'ReliefWeb'
  }
];

// Filter keywords STRICTLY related to NATURAL DISASTERS in the Philippines
const DISASTER_KEYWORDS = [
  // Typhoons & Storms
  'bagyo', 'typhoon', 'storm', 'cyclone', 'hurricane', 'habagat', 'monsoon',
  'signal no. 1', 'signal no. 2', 'signal no. 3', 'signal no. 4', 'signal no. 5',
  
  // Flooding & Rain
  'flood', 'baha', 'tubig', 'binaha', 'heavy rain', 'malakas na ulan', 'bumuhos', 
  'pag-ulan', 'naapektuhan ng baha', 'flashflood', 'rising water level',
  
  // Landslides
  'landslide', 'pagguho', 'guho', 'mudslide', 'erosion', 'soil erosion', 
  'nakabaon', 'nadaganan', 'gumuho ang lupa',
  
  // Earthquakes
  'earthquake', 'lindol', 'magnitude', 'intensity', 'aftershock', 'tremor',
  'lumindol', 'yumanig', 'phivolcs', 'fault line', 'epicenter', 'seismic',
  
  // Tsunamis
  'tsunami', 'tidal wave', 'storm surge', 'sea level rise', 'coastal flooding',
  
  // Volcanic Activity
  'volcanic', 'bulkan', 'volcano', 'ash fall', 'pyroclastic flow', 'lava', 
  'eruption', 'pumutok', 'taal', 'mayon', 'kanlaon', 'bulusan', 'alert level',
  
  // Drought & El Niño
  'drought', 'tagtuyot', 'el nino', 'el niño', 'water shortage', 'kakulangan ng tubig',
  'dry spell', 'crop failure', 'kakapusan ng tubig',
  
  // Disaster Response Terms
  'evacuated', 'evacuation', 'evacuees', 'rescue', 'nasalanta', 'stranded',
  'relief', 'casualties', 'fatalities', 'injured', 'missing', 'displaced',
  'destroyed homes', 'damages', 'relief goods', 'relief operations',
  
  // Government Agencies
  'ndrrmc', 'pagasa', 'phivolcs', 'disaster agency', 'ocd', 'red cross',
  
  // Warning Levels
  'red alert', 'orange alert', 'warning', 'disaster', 'calamity', 'state of calamity'
];

/**
 * Adds a WebSocket client to the list of connected clients
 */
export function addWebSocketClient(ws: WebSocket) {
  connectedClients.add(ws);
  
  // Remove the client when it disconnects
  ws.on('close', () => {
    connectedClients.delete(ws);
    log(`Real-time feed client disconnected, ${connectedClients.size} remaining`, 'real-news');
  });
  
  log(`New real-time feed client connected, total clients: ${connectedClients.size}`, 'real-news');
}

/**
 * Broadcasts new posts to all connected WebSocket clients
 */
export function broadcastNewPost(post: any) {
  try {
    const messageData = {
      type: 'new_post',
      data: post,
      timestamp: new Date().toISOString()
    };
    
    connectedClients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(messageData));
      }
    });
    
    log(`Broadcasted new post to ${connectedClients.size} clients`, 'real-news');
  } catch (error) {
    log(`Error broadcasting new post: ${error}`, 'real-news');
  }
}

/**
 * Checks if text contains disaster-related keywords and qualifies as a natural disaster post
 */
function containsDisasterKeywords(text: string): boolean {
  if (!text) return false;
  
  const lowerText = text.toLowerCase();
  
  // First check: Basic keyword matching
  const hasKeyword = DISASTER_KEYWORDS.some(keyword => lowerText.includes(keyword.toLowerCase()));
  
  if (!hasKeyword) return false;
  
  // Second check: Must not contain terms that indicate non-natural disasters
  const nonNaturalDisasterTerms = [
    'terrorist', 'terrorism', 'bombing', 'shooter', 'shooting', 'hostage', 'kidnap',
    'attack', 'war', 'coup', 'protest', 'rally', 'demonstration', 'riot',
    'covid', 'pandemic', 'virus', 'lockdown', 'quarantine', 'omicron', 'delta',
    'crime', 'murder', 'homicide', 'rape', 'carnap', 'carjacking', 'robbery',
    'scandal', 'corruption', 'graft', 'investigation'
  ];
  
  // If it has any non-natural disaster terms, return false
  if (nonNaturalDisasterTerms.some(term => lowerText.includes(term))) {
    return false;
  }
  
  // Third check: Must have at least one term that strongly indicates natural disaster
  const strongNaturalDisasterIndicators = [
    'typhoon', 'bagyo', 'flood', 'baha', 'landslide', 'guho', 'earthquake', 'lindol',
    'volcanic', 'eruption', 'tsunami', 'storm', 'monsoon', 'habagat', 'cyclone',
    'drought', 'el nino', 'el niño', 'forest fire', 'wildfire', 'sunog', 'magnitude',
    'pagasa', 'phivolcs', 'ndrrmc', 'evacuate', 'evacuees', 'evacuation'
  ];
  
  return strongNaturalDisasterIndicators.some(term => lowerText.includes(term));
}

/**
 * Fetches news from a single source
 */
async function fetchNewsFromSource(source: typeof NEWS_SOURCES[0]): Promise<any[]> {
  try {
    log(`Fetching news from ${source.name}...`, 'real-news');
    
    const feed = await parser.parseURL(source.url);
    const disasterRelatedItems = feed.items.filter(item => 
      containsDisasterKeywords(item.title || '') || 
      containsDisasterKeywords(item.content || '') ||
      containsDisasterKeywords(item.contentSnippet || '')
    );
    
    log(`Found ${disasterRelatedItems.length} disaster-related items from ${source.name}`, 'real-news');
    return disasterRelatedItems.map(item => ({
      ...item,
      sourceName: source.source
    }));
  } catch (error) {
    log(`Error fetching from ${source.name}: ${error}`, 'real-news');
    return [];
  }
}

/**
 * Processes a news item and saves it to the database
 */
async function processNewsItem(item: any): Promise<any> {
  try {
    // Extract text from the item
    const title = item.title || '';
    const content = item.contentSnippet || item.content || '';
    const postText = `${title}. ${content}`.substring(0, 1000); // Limit length
    
    // Check if we already have this item in our database (by title)
    const existingPosts = await storage.getSentimentPosts();
    const alreadyExists = existingPosts.some(post => 
      post.text.includes(title) || title.includes(post.text)
    );
    
    if (alreadyExists) {
      log(`Skipping already existing news item: ${title}`, 'real-news');
      return null;
    }
    
    // Analyze the sentiment of the news item
    const result = await pythonService.analyzeSentiment(postText);
    
    // Create and save the post
    const newPost = await storage.createSentimentPost({
      text: postText,
      sentiment: result.sentiment,
      confidence: result.confidence,
      source: item.sourceName || "Philippine News",
      language: result.language || "en",
      location: result.location || null,
      disasterType: result.disasterType || null,
      explanation: result.explanation
    });
    
    // Get the complete post with ID from the database
    const savedPost = await storage.getSentimentPostById(newPost.id);
    
    // Broadcast the new post to all WebSocket clients
    if (savedPost) {
      broadcastNewPost(savedPost);
    }
    
    log(`Processed new news item: "${title}"`, 'real-news');
    return savedPost;
  } catch (error) {
    log(`Error processing news item: ${error}`, 'real-news');
    return null;
  }
}

/**
 * Fetches news from all sources
 */
export async function fetchAllNews(): Promise<any[]> {
  try {
    // Fetch from all sources in parallel
    const allNewsPromises = NEWS_SOURCES.map(source => fetchNewsFromSource(source));
    const allNewsResults = await Promise.all(allNewsPromises);
    
    // Flatten results
    const allNews = allNewsResults.flat();
    
    // Sort by latest first
    allNews.sort((a, b) => {
      const dateA = a.isoDate || a.pubDate || new Date();
      const dateB = b.isoDate || b.pubDate || new Date();
      return new Date(dateB).getTime() - new Date(dateA).getTime();
    });
    
    log(`Fetched a total of ${allNews.length} disaster-related news items`, 'real-news');
    return allNews;
  } catch (error) {
    log(`Error fetching all news: ${error}`, 'real-news');
    return [];
  }
}

/**
 * Processes all fetched news items
 */
async function processAllNews(): Promise<void> {
  try {
    // Fetch all news items
    const allNews = await fetchAllNews();
    
    // Take only the top 5 most recent items to avoid flooding
    const recentNews = allNews.slice(0, 5);
    
    // Process items sequentially to avoid overwhelming the system
    for (const item of recentNews) {
      await processNewsItem(item);
      // Add a small delay between processing
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    log(`Completed processing ${recentNews.length} news items`, 'real-news');
  } catch (error) {
    log(`Error in processAllNews: ${error}`, 'real-news');
  }
}

// Track the timer reference so we can restart it if needed
let newsFeedInterval: NodeJS.Timeout | null = null;

/**
 * Starts fetching real-time news
 */
export function startRealNewsFeed(): void {
  // Stop existing timer if running
  stopRealNewsFeed();
  
  // Process immediately
  processAllNews();
  
  // Set interval to fetch news every 15 minutes
  newsFeedInterval = setInterval(processAllNews, 15 * 60 * 1000);
  
  log('Real news feed started successfully', 'real-news');
}

/**
 * Stops fetching real-time news
 */
export function stopRealNewsFeed(): void {
  if (newsFeedInterval) {
    clearInterval(newsFeedInterval);
    newsFeedInterval = null;
    log('Real news feed stopped', 'real-news');
  }
}

/**
 * Gets the most recent posts for the real-time feed
 */
export async function getLatestPosts(limit: number = 20) {
  try {
    return await storage.getRecentSentimentPosts(limit);
  } catch (error) {
    log(`Error getting latest posts: ${error}`, 'real-news');
    return [];
  }
}

/**
 * Manually triggers fetching news right now
 */
export async function manuallyFetchNews(): Promise<any[]> {
  log('Manually fetching news', 'real-news');
  await processAllNews();
  return getLatestPosts();
}