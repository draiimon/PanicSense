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
  // Mga pangunahing news sources na gumagana
  {
    name: 'PhilStar Nation',
    url: 'https://www.philstar.com/rss/nation',
    source: 'PhilStar'
  },
  {
    name: 'Inquirer Feed',
    url: 'https://www.inquirer.net/fullfeed',
    source: 'Inquirer'
  },
  {
    name: 'Manila Times',
    url: 'https://www.manilatimes.net/news/feed',
    source: 'Manila Times'
  },
  {
    name: 'Manila Standard',
    url: 'https://manilastandard.net/feed/',
    source: 'Manila Standard'
  },
  {
    name: 'BusinessWorld',
    url: 'https://www.bworldonline.com/feed/',
    source: 'BusinessWorld'
  },
  // Dagdag na malalaking news sources
  {
    name: 'ABS-CBN News',
    url: 'https://news.abs-cbn.com/rss/news',
    source: 'ABS-CBN News'
  },
  {
    name: 'Rappler',
    url: 'https://www.rappler.com/nation/feed/',
    source: 'Rappler'
  },
  // Weather and Disaster agencies
  {
    name: 'PAGASA News',
    url: 'https://bagong.pagasa.dost.gov.ph/press-release-archive/?format=feed&type=rss',
    source: 'PAGASA'
  },
  
  // Malalaking news agencies sa Pilipinas
  {
    name: 'GMA News',
    url: 'https://www.gmanetwork.com/news/rss',
    source: 'GMA News'
  },
  // Removed duplicate ABS-CBN entry and error comment since we have a working one above
  // Manila Bulletin returning error - removed
  {
    name: 'PNA',
    url: 'https://pna.gov.ph/rss',
    source: 'Philippine News Agency'
  },
  
  // Regional news sources
  // SunStar returning 404 error
  /*{
    name: 'SunStar Philippines',
    url: 'https://www.sunstar.com.ph/rss',
    source: 'SunStar'
  },*/
  {
    name: 'Cebu Daily News',
    url: 'https://cebudailynews.inquirer.net/feed',
    source: 'Cebu Daily News'
  },
  {
    name: 'Mindanao Times',
    url: 'https://mindanaotimes.com.ph/feed',
    source: 'Mindanao Times'
  },
  {
    name: 'Panay News',
    url: 'https://www.panaynews.net/feed',
    source: 'Panay News'
  },
  {
    name: 'Bohol Chronicle',
    url: 'https://boholchronicle.com.ph/feed',
    source: 'Bohol Chronicle'
  },
  {
    name: 'Punto Central Luzon',
    url: 'https://punto.com.ph/feed',
    source: 'Punto Central Luzon'
  },
  {
    name: 'Journal Online',
    url: 'https://journal.com.ph/feed',
    source: 'Journal Online'
  },
  {
    name: 'Metro Cebu News',
    url: 'https://metrocebu.news/feed',
    source: 'Metro Cebu News'
  },
  {
    name: 'Baguio Midland Courier',
    url: 'https://baguiomidlandcourier.com.ph/feed',
    source: 'Baguio Midland Courier'
  },
  
  // Mga internacional at disaster-specific feeds
  {
    name: 'ReliefWeb Philippines',
    url: 'https://reliefweb.int/updates/rss?search=philippines',
    source: 'ReliefWeb'
  }
  
  // Mga karagdagang feed ay sinubukan pero hindi compatible sa kasalukuyang RSS parser
  // Kaya kinomento muna para hindi maantala ang mga gumaganang feed
  /*
  {
    name: 'PAGASA Forecasts',
    url: 'https://bagong.pagasa.dost.gov.ph/rss',
    source: 'PAGASA'
  },
  {
    name: 'GDACS Global Disasters',
    url: 'https://www.gdacs.org/rss.aspx',
    source: 'GDACS'
  }
  */
];

// Filter keywords STRICTLY related to NATURAL DISASTERS in the Philippines
const DISASTER_KEYWORDS = [
  // Typhoons & Storms
  'bagyo', 'typhoon', 'storm', 'cyclone', 'hurricane', 'habagat', 'monsoon',
  'signal no. 1', 'signal no. 2', 'signal no. 3', 'signal no. 4', 'signal no. 5',
  'tropical depression', 'low pressure area', 'amihan', 'malakas na hangin',
  
  // Flooding & Rain
  'flood', 'baha', 'tubig', 'binaha', 'heavy rain', 'malakas na ulan', 'bumuhos', 
  'pag-ulan', 'naapektuhan ng baha', 'flashflood', 'rising water level',
  'high tide', 'tumataas na tubig', 'tubig-baha', 'nabaha',
  
  // Landslides
  'landslide', 'pagguho', 'guho', 'mudslide', 'erosion', 'soil erosion', 
  'nakabaon', 'nadaganan', 'gumuho ang lupa', 'pagguho ng lupa', 'landslip',
  
  // Earthquakes
  'earthquake', 'lindol', 'magnitude', 'intensity', 'aftershock', 'tremor',
  'lumindol', 'yumanig', 'phivolcs', 'fault line', 'epicenter', 'seismic',
  'ground shaking', 'quake', 'temblor',
  
  // Tsunamis
  'tsunami', 'tidal wave', 'storm surge', 'sea level rise', 'coastal flooding',
  'daluyong', 'alat', 'alon', 'tubig-dagat',
  
  // Volcanic Activity
  'volcanic', 'bulkan', 'volcano', 'ash fall', 'pyroclastic flow', 'lava', 
  'eruption', 'pumutok', 'taal', 'mayon', 'kanlaon', 'bulusan', 'alert level',
  'phreatic', 'magma', 'abo', 'ashfall', 'lahar',
  
  // Drought & El Niño / Extreme Heat
  'drought', 'tagtuyot', 'el nino', 'el niño', 'water shortage', 'kakulangan ng tubig',
  'dry spell', 'crop failure', 'kakapusan ng tubig', 'heatwave', 'heat index',
  'heat stroke', 'extreme heat', 'matinding init', 'nakamamatay na init',
  
  // Disaster Response Terms
  'evacuated', 'evacuation', 'evacuees', 'rescue', 'nasalanta', 'stranded',
  'relief', 'casualties', 'fatalities', 'injured', 'missing', 'displaced',
  'destroyed homes', 'damages', 'relief goods', 'relief operations',
  'emergency shelter', 'relief center', 'evacuation center',
  
  // Government Agencies
  'ndrrmc', 'pagasa', 'phivolcs', 'disaster agency', 'ocd', 'red cross',
  'disaster response', 'disaster management', 'LGU disaster', 'DOST disaster',
  
  // Warning Levels
  'red alert', 'orange alert', 'warning', 'disaster', 'calamity', 'state of calamity',
  'weather alert', 'weather warning', 'weather advisory'
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
    'pagasa', 'phivolcs', 'ndrrmc', 'evacuate', 'evacuees', 'evacuation',
    'heatwave', 'heat index', 'heat stroke', 'extreme heat', 'matinding init',
    'alert level', 'disaster', 'calamity', 'rising water', 'rising sea'
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
    let content = item.contentSnippet || item.content || '';

    // Truncate excessively long content to prevent processing issues
    // This is particularly important for Manila Times and other sources that include full article text
    if (content.length > 800) {
      content = content.substring(0, 800) + '...';
      log(`Truncated long content from "${item.sourceName || 'unknown source'}" to 800 chars`, 'real-news');
    }
    
    // Create a clean post text combining title and truncated content
    const postText = `${title}. ${content}`;
    
    // Enhanced duplicate check: Check if we already have this item in our database
    // 1. First check by exact title match
    const existingPosts = await storage.getSentimentPosts();
    
    // Check for duplicate by exact title match (most reliable)
    if (existingPosts.some(post => post.text.startsWith(title) || title === post.text)) {
      log(`Skipping duplicate news item (exact title match): ${title}`, 'real-news');
      return null;
    }
    
    // Check for partial title match (for slightly modified titles)
    const titleWords = title.toLowerCase().split(/\s+/).filter((w: string) => w.length > 3);
    if (titleWords.length > 3) {  // Only check if we have enough significant words
      const titleMatches = existingPosts.filter(post => {
        const postTitlePart = post.text.split('.')[0].toLowerCase();
        return titleWords.filter((word: string) => postTitlePart.includes(word)).length >= Math.min(3, titleWords.length * 0.7);
      });
      
      if (titleMatches.length > 0) {
        log(`Skipping likely duplicate news item (partial title match): ${title}`, 'real-news');
        return null;
      }
    }
    
    // Analyze the sentiment of the news item
    const result = await pythonService.analyzeSentiment(postText);
    
    // Get actual timestamp from post if available
    const postTimestamp = item.isoDate || item.pubDate ? new Date(item.isoDate || item.pubDate) : new Date();
    
    // Detect actual disaster type - if we have a clear disaster type, use it
    // Otherwise, rely on the python service's analysis
    const detectedDisasterType = pythonService.extractDisasterTypeFromText(postText);
    const finalDisasterType = detectedDisasterType || 
      ((result.disasterType === "Unknown Disaster" || !result.disasterType) ? "UNKNOWN" : result.disasterType);
    
    // Detect a single specific location rather than listing all mentioned locations
    // Use the result.location from Python service first as it may have better AI-based detection
    // If that's not available, try the JS-based extraction as a fallback
    const finalLocation = result.location || 
      (typeof result.location === 'string' && result.location !== 'UNKNOWN' ? result.location : null) || 
      "UNKNOWN";
    
    // Create and save the post with improved metadata
    const newPost = await storage.createSentimentPost({
      text: postText,
      sentiment: result.sentiment,
      confidence: result.confidence,
      source: item.sourceName || "Philippine News",
      language: result.language || "en",
      location: finalLocation,
      disasterType: finalDisasterType,
      explanation: result.explanation,
      timestamp: postTimestamp
    });
    
    // Get the complete post with ID from the database
    const savedPost = await storage.getSentimentPostById(newPost.id);
    
    // Broadcast the new post to all WebSocket clients
    if (savedPost) {
      broadcastNewPost(savedPost);
    }
    
    log(`Processed new news item: "${title}" [Type: ${finalDisasterType}, Location: ${finalLocation}]`, 'real-news');
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
 * Processes all fetched news items with improved rate limiting
 */
async function processAllNews(): Promise<void> {
  try {
    // Fetch all news items
    const allNews = await fetchAllNews();
    
    // Handle empty results gracefully
    if (!allNews || allNews.length === 0) {
      log('No news items to process or all sources returned empty results', 'real-news');
      return;
    }
    
    // Take only the top 3 most recent items to avoid API rate limits
    // This is critical for ensuring we don't hit API limits while still getting fresh content
    const recentNews = allNews.slice(0, 3);
    
    log(`Processing ${recentNews.length} most recent news items out of ${allNews.length} total`, 'real-news');
    
    // Track which sources we've already processed to avoid duplicate processing
    const processedSources = new Set<string>();
    
    // Prioritize news items from more diverse sources
    // Sort by source first to ensure we get news from different sources
    const diverseRecentNews = recentNews.sort((a, b) => {
      // First prioritize items from sources we haven't processed yet
      const aProcessed = processedSources.has(a.sourceName || '');
      const bProcessed = processedSources.has(b.sourceName || '');
      
      if (aProcessed !== bProcessed) {
        return aProcessed ? 1 : -1; // Place unprocessed sources first
      }
      
      // Then prioritize by date
      const dateA = a.isoDate || a.pubDate || new Date();
      const dateB = b.isoDate || b.pubDate || new Date();
      return new Date(dateB).getTime() - new Date(dateA).getTime();
    });
    
    // Process items sequentially with improved error handling and longer delays
    for (const item of diverseRecentNews) {
      try {
        const source = item.sourceName || 'unknown';
        
        // Skip if we've already processed an item from this source in this batch
        if (processedSources.has(source)) {
          log(`Skipping additional item from already processed source: ${source}`, 'real-news');
          continue;
        }
        
        // Process the item
        const result = await processNewsItem(item);
        
        // Mark this source as processed
        processedSources.add(source);
        
        // If successfully processed, add a longer delay (3 seconds) between sources
        // This ensures we don't overwhelm any APIs and helps prevent rate limiting
        if (result) {
          await new Promise(resolve => setTimeout(resolve, 3000));
        } else {
          // Shorter delay if item was skipped (e.g., duplicate)
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      } catch (itemError) {
        log(`Error processing individual news item: ${itemError}`, 'real-news');
        // Continue with next item even if one fails
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    log(`Completed processing news items from ${processedSources.size} unique sources`, 'real-news');
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
  
  // Set interval to fetch news every 10 minutes to increase fresh content
  newsFeedInterval = setInterval(processAllNews, 10 * 60 * 1000);
  
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