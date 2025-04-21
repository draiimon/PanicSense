import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { pythonService } from '../python-service';
import { v4 as uuidv4 } from 'uuid';

// Get the current module's directory (for ESM)
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Define interfaces
export interface SocialMediaPost {
  id: string;
  content: string;
  date: string;
  source: string;
  user: {
    username: string;
    displayname: string;
    verified: boolean;
  };
  hashtags: string[];
  location?: string;
  url: string;
}

// Philippine disaster-related hashtags to monitor
const DISASTER_HASHTAGS = [
  // Weather/Typhoon related
  "BagyoPH",       // Typhoon
  "TyphoonPH",     // Typhoon (English)
  "HabagaPH",      // Monsoon
  
  // Earthquake related  
  "LindolPH",      // Earthquake
  "EarthquakePH",  // Earthquake (English)
  
  // Flood related
  "BahaPH",        // Flood
  "FloodPH",       // Flood (English)
  "DelubiyoPH",    // Flood (biblical term)
  
  // Fire related
  "SunogPH",       // Fire
  "SunoqPH",       // Fire (alternative spelling)
  "FirePH",        // Fire (English)
  
  // General disaster/relief terms
  "SakunaPH",      // General disaster term
  "EmergencyPH",   // Emergency
  "AlertPH",       // Alert
  "EvacuatePH",    // Evacuation
  "RescuePH",      // Rescue operations
  "ReliefPH"       // Relief efforts
];

// Official disaster management accounts
const OFFICIAL_ACCOUNTS = [
  "phivolcs_dost",   // PHIVOLCS
  "dost_pagasa",     // PAGASA
  "ndrrmc_opcen",    // NDRRMC
  "philredcross"     // Philippine Red Cross
];

export class SocialMediaMonitor {
  private posts: SocialMediaPost[];
  private lastFetched: Date;
  private fetchInterval: number; // in milliseconds
  private fetchInProgress: boolean;

  constructor() {
    this.posts = [];
    this.lastFetched = new Date(0); // Begin with earliest date
    this.fetchInterval = 5 * 60 * 1000; // 5 minutes
    this.fetchInProgress = false;
    
    // Start fetching immediately and then regularly
    this.fetchSocialMediaPosts();
    setInterval(() => this.fetchSocialMediaPosts(), this.fetchInterval);
  }

  /**
   * Fetch social media posts related to disasters in the Philippines
   */
  private async fetchSocialMediaPosts(): Promise<void> {
    // Prevent multiple concurrent fetches
    if (this.fetchInProgress) {
      return;
    }
    
    this.fetchInProgress = true;
    
    try {
      console.log('[social-media] Fetching tweets for disaster hashtags...');
      
      // Use the Python scraper to get tweets
      const result = await this.runPythonScraper();
      
      if (result && Array.isArray(result) && result.length > 0) {
        // Replace the existing posts with the new ones
        this.posts = result;
        console.log(`[social-media] Fetched a total of ${result.length} disaster-related tweets`);
      } else {
        console.log('[social-media] No tweets to process');
        // Keep existing posts if we didn't get any new ones
      }
      
      this.lastFetched = new Date();
    } catch (error) {
      console.error('[social-media] Social media scraper error:', error);
    } finally {
      this.fetchInProgress = false;
    }
  }

  /**
   * Run the Python scraper to fetch tweets
   */
  private runPythonScraper(): Promise<SocialMediaPost[]> {
    return new Promise((resolve, reject) => {
      const scriptPath = path.join(__dirname, '../python/social_media_scraper.py');
      
      // Join hashtags and limit to a reasonable number to avoid overloading the scraper
      const hashtags = DISASTER_HASHTAGS.slice(0, 10).join(',');
      
      // Number of tweets to fetch per hashtag
      const tweetsPerHashtag = 3;
      
      const pythonProcess = spawn('python3', [scriptPath, hashtags, tweetsPerHashtag.toString()]);
      
      let dataString = '';
      
      pythonProcess.stdout.on('data', (data) => {
        dataString += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        console.error('[social-media] Social media scraper error:', data.toString());
      });
      
      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error(`[social-media] Python scraper exited with code ${code}`);
          // If there's a problem with the scraper, provide at least some fallback data
          // from our last successful fetch
          if (this.posts.length > 0) {
            resolve(this.posts);
          } else {
            // If we've never successfully fetched, resolve with an empty array
            resolve([]);
          }
          return;
        }
        
        try {
          // Parse the JSON output from the Python script
          const tweets = JSON.parse(dataString);
          // Make sure we have proper SocialMediaPost objects
          resolve(this.validateAndProcessTweets(tweets));
        } catch (error) {
          console.error('[social-media] Error parsing Python output:', error);
          resolve(this.posts); // Return existing posts on error
        }
      });
    });
  }

  /**
   * Validate and process the tweets from the Python scraper
   */
  private validateAndProcessTweets(tweets: any[]): SocialMediaPost[] {
    if (!Array.isArray(tweets)) {
      return [];
    }
    
    return tweets.map(tweet => {
      // Ensure we have a valid tweet object
      const post: SocialMediaPost = {
        id: tweet.id || uuidv4(),
        content: tweet.content || tweet.renderedContent || 'No content available',
        date: tweet.date || new Date().toISOString(),
        source: 'Twitter/X',
        user: {
          username: tweet.user?.username || 'unknown',
          displayname: tweet.user?.displayname || 'Unknown User',
          verified: OFFICIAL_ACCOUNTS.includes(tweet.user?.username) || false
        },
        hashtags: Array.isArray(tweet.hashtags) ? tweet.hashtags : [],
        location: tweet.location || 'Philippines',
        url: tweet.url || `https://twitter.com/unknown/status/${tweet.id || 'unknown'}`
      };
      
      return post;
    });
  }

  /**
   * Get the latest social media posts, sorted by date (newest first)
   */
  public async getLatestPosts(): Promise<SocialMediaPost[]> {
    // If data is older than the fetch interval, refresh it
    const currentTime = new Date();
    if (currentTime.getTime() - this.lastFetched.getTime() > this.fetchInterval) {
      await this.fetchSocialMediaPosts();
    }
    
    // Sort by date (newest first)
    return [...this.posts].sort((a, b) => 
      new Date(b.date).getTime() - new Date(a.date).getTime()
    );
  }
}

// Initialize and export the monitor
const socialMediaMonitor = new SocialMediaMonitor();

export function startSocialMediaMonitor(): void {
  // Manually trigger a fetch to start the monitoring
  socialMediaMonitor.getLatestPosts()
    .then(() => console.log('[social-media] Social media monitoring started successfully'))
    .catch(err => console.error('[social-media] Error starting social media monitoring:', err));
}

export function stopSocialMediaMonitor(): void {
  // This function is a placeholder in case we need to implement cleanup in the future
  console.log('[social-media] Social media monitoring stopped');
}

export default socialMediaMonitor;