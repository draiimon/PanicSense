import axios from 'axios';
import Parser from 'rss-parser';
import { v4 as uuidv4 } from 'uuid';

// Define the RSS parser
const parser = new Parser({
  customFields: {
    item: [
      ['media:content', 'media'],
      ['description', 'description'],
      ['content:encoded', 'contentEncoded']
    ]
  }
});

// News item interface
export interface NewsItem {
  id: string;
  title: string;
  content: string;
  source: string;
  timestamp: string;
  url: string;
  disasterType?: string;
  location?: string;
}

// Philippines disaster keywords for filtering relevant news
const DISASTER_KEYWORDS = [
  // Tagalog terms
  'bagyo', 'lindol', 'baha', 'sunog', 'sakuna', 'kalamidad', 'pagsabog', 'bulkan',
  'pagputok', 'guho', 'tagtuyot', 'init', 'pagguho', 'habagat', 'pinsala', 'tsunami',
  'salanta', 'ulan', 'dagundong', 'likas', 'evacuate', 'evacuation',
  
  // English terms
  'typhoon', 'earthquake', 'flood', 'fire', 'disaster', 'calamity', 'eruption', 'volcano',
  'landslide', 'drought', 'heat wave', 'tsunami', 'storm', 'damage', 'tremor', 'aftershock',
  'evacuation', 'emergency', 'relief', 'rescue', 'warning', 'alert', 'NDRRMC', 'PAGASA', 'PHIVOLCS'
];

// Philippine location keywords for detecting affected areas
const LOCATION_KEYWORDS = [
  'Manila', 'Quezon City', 'Cebu', 'Davao', 'Luzon', 'Visayas', 'Mindanao',
  'Cavite', 'Laguna', 'Batangas', 'Rizal', 'Bulacan', 'Pampanga', 'Bicol',
  'Leyte', 'Samar', 'Iloilo', 'Negros', 'Zambales', 'Pangasinan', 'Bataan',
  'Nueva Ecija', 'Cagayan', 'Palawan', 'Baguio', 'Tacloban', 'Cotabato',
  'Zamboanga', 'Albay', 'Sorsogon', 'Marinduque', 'Aklan', 'Capiz', 'Antique'
];

// Disaster type classification based on keywords
const DISASTER_TYPE_KEYWORDS: Record<string, string[]> = {
  'typhoon': ['bagyo', 'typhoon', 'storm', 'cyclone', 'hurricane', 'habagat', 'monsoon', 'salanta'],
  'earthquake': ['lindol', 'earthquake', 'tremor', 'aftershock', 'temblor', 'yugyug', 'dagundong'],
  'flood': ['baha', 'flood', 'flooding', 'flash flood', 'delubyo', 'pagbaha'],
  'fire': ['sunog', 'fire', 'blaze', 'flames', 'burning', 'arson'],
  'volcano': ['bulkan', 'volcano', 'volcanic', 'eruption', 'ashfall', 'lahar', 'pagputok'],
  'landslide': ['guho', 'landslide', 'mudslide', 'rockfall', 'pagguho', 'tabon'],
  'extreme heat': ['init', 'heat wave', 'extreme heat', 'high temperature', 'tagtuyot'],
  'drought': ['tagtuyot', 'drought', 'dry spell', 'water shortage', 'El Niño'],
  'tsunami': ['tsunami', 'tidal wave', 'alon bundol'],
};

export class RealNewsService {
  private newsSources: {url: string, name: string}[];
  private cachedNews: NewsItem[];
  private lastFetched: Date;
  private fetchInterval: number; // in milliseconds

  constructor() {
    // Initialize with Philippine news sources that have RSS feeds (verified working)
    this.newsSources = [
      // Working news sources (verified)
      { name: 'Manila Times', url: 'https://www.manilatimes.net/news/feed/' },
      { name: 'BusinessWorld', url: 'https://www.bworldonline.com/feed/' },
      { name: 'Rappler', url: 'https://www.rappler.com/feed/' },
      { name: 'Cebu Daily News', url: 'https://cebudailynews.inquirer.net/feed' },
      { name: 'Panay News', url: 'https://www.panaynews.net/feed/' },
      { name: 'Mindanao Times', url: 'https://mindanaotimes.com.ph/feed/' },
      
      // National news sources
      { name: 'PhilStar Headlines', url: 'https://www.philstar.com/rss/headlines' },
      { name: 'PhilStar Nation', url: 'https://www.philstar.com/rss/nation' },
      { name: 'NewsInfo Inquirer', url: 'https://newsinfo.inquirer.net/feed' },
      { name: 'Manila Bulletin', url: 'https://mb.com.ph/feed/' },
      { name: 'Tribune News', url: 'https://prod-qt-images.s3.amazonaws.com/production/tribune/feed.xml' },
      
      // Regional news sources - Luzon
      { name: 'PhilStar Metro', url: 'https://www.philstar.com/rss/metro' },
      { name: 'PhilStar Luzon', url: 'https://www.philstar.com/rss/region/luzon' },
      
      // Regional news sources - Visayas
      { name: 'PhilStar Visayas', url: 'https://www.philstar.com/rss/region/visayas' },
      { name: 'SunStar Cebu', url: 'https://www.sunstar.com.ph/rss?id=4' },
      
      // Regional news sources - Mindanao
      { name: 'PhilStar Mindanao', url: 'https://www.philstar.com/rss/region/mindanao' },
      { name: 'MindaNews', url: 'https://www.mindanews.com/feed/' }
    ];
    
    this.cachedNews = [];
    this.lastFetched = new Date(0); // Begin with earliest date
    this.fetchInterval = 10 * 60 * 1000; // 10 minutes
    
    // Start fetching news immediately and then regularly
    this.fetchAllNews();
    setInterval(() => this.fetchAllNews(), this.fetchInterval);
  }

  /**
   * Fetch all news from configured sources
   */
  private async fetchAllNews(): Promise<void> {
    const allNews: NewsItem[] = [];
    
    // Fetch from all sources in parallel
    const fetchPromises = this.newsSources.map(source => this.fetchFromSource(source));
    const results = await Promise.allSettled(fetchPromises);
    
    // Process the results
    results.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        const newsItems = result.value;
        if (newsItems && newsItems.length > 0) {
          console.log(`[real-news] Found ${newsItems.length} disaster-related items from ${this.newsSources[index].name}`);
          allNews.push(...newsItems);
        } else {
          console.log(`[real-news] Found 0 disaster-related items from ${this.newsSources[index].name}`);
        }
      } else {
        console.log(`[real-news] Error fetching from ${this.newsSources[index].name}: ${result.reason}`);
      }
    });
    
    // Update our cache
    this.cachedNews = allNews;
    this.lastFetched = new Date();
  }

  /**
   * Fetch news from a specific source
   */
  private async fetchFromSource(source: {url: string, name: string}): Promise<NewsItem[]> {
    try {
      // Fetch the RSS feed with a timeout
      const response = await axios.get(source.url, { timeout: 10000 });
      
      // Parse the feed
      const feed = await parser.parseString(response.data);
      
      if (!feed.items || feed.items.length === 0) {
        return [];
      }
      
      // Filter for disaster-related news and transform to our format
      const newsItems: NewsItem[] = feed.items
        .filter(item => this.isDisasterRelated(item.title || '', (item.contentSnippet || item.content || '')))
        .map(item => {
          // Extract best content from item
          const content = item.contentEncoded || 
                        item.content || 
                        item.contentSnippet || 
                        item.description || 
                        'No content available';
          
          // Clean the content (remove HTML)
          const cleanContent = this.stripHtml(content);
          
          return {
            id: item.guid || uuidv4(),
            title: item.title || 'No title',
            content: cleanContent.substring(0, 500) + (cleanContent.length > 500 ? '...' : ''),
            source: source.name,
            timestamp: item.isoDate || new Date().toISOString(),
            url: item.link || '',
            disasterType: this.classifyDisasterType(item.title || '', cleanContent),
            location: this.extractLocation(item.title || '', cleanContent)
          };
        });
      
      return newsItems;
    } catch (error) {
      // Throw error with context
      throw error;
    }
  }

  /**
   * Check if an article is disaster-related based on keywords
   */
  private isDisasterRelated(title: string, content: string): boolean {
    if (!title && !content) return false;
    
    const combinedText = `${title} ${content}`.toLowerCase();
    
    return DISASTER_KEYWORDS.some(keyword => combinedText.includes(keyword.toLowerCase()));
  }

  /**
   * Classify the type of disaster mentioned in the article
   */
  private classifyDisasterType(title: string, content: string): string {
    if (!title && !content) return '';
    
    const combinedText = `${title} ${content}`.toLowerCase();
    
    // Check each disaster type
    for (const [disasterType, keywords] of Object.entries(DISASTER_TYPE_KEYWORDS)) {
      if (keywords.some(keyword => combinedText.includes(keyword.toLowerCase()))) {
        return disasterType;
      }
    }
    
    // If no specific disaster type detected but it passed the disaster filter,
    // it's a general disaster update
    return 'disaster update';
  }

  /**
   * Extract location information from the article
   */
  private extractLocation(title: string, content: string): string {
    if (!title && !content) return 'Philippines';
    
    const combinedText = `${title} ${content}`;
    
    // Check for mentions of specific locations in the Philippines
    for (const location of LOCATION_KEYWORDS) {
      if (combinedText.includes(location)) {
        return location;
      }
    }
    
    // Default to Philippines if no specific location is found
    return 'Philippines';
  }

  /**
   * Strip HTML tags from content
   */
  private stripHtml(html: string): string {
    // Simple HTML stripping - in production you might want a more robust solution
    return html.replace(/<[^>]*>?/gm, ' ')
               .replace(/\s\s+/g, ' ')
               .trim();
  }

  /**
   * Get the latest news, sorted by date (newest first)
   */
  public async getLatestNews(): Promise<NewsItem[]> {
    // If data is older than the fetch interval, refresh it
    const currentTime = new Date();
    if (currentTime.getTime() - this.lastFetched.getTime() > this.fetchInterval) {
      await this.fetchAllNews();
    }
    
    // Sort by timestamp (newest first)
    return [...this.cachedNews].sort((a, b) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }
}