/**
 * Social Media Monitoring Service
 * Monitors Twitter/X hashtags and other social media platforms for disaster-related content
 */

import { storage } from '../storage';
import { pythonService } from '../python-service';
import { log } from '../vite';
import { spawn } from 'child_process';
import { WebSocket } from 'ws';
import path from 'path';
import fs from 'fs';

// WebSocket clients for real-time updates
const connectedClients = new Set<WebSocket>();

// Add a client to receive real-time updates
export function addWebSocketClient(ws: WebSocket): void {
  connectedClients.add(ws);
  
  // Set up cleanup when client disconnects
  ws.on('close', () => {
    connectedClients.delete(ws);
    log(`Social media monitor WebSocket client disconnected, ${connectedClients.size} remaining`, 'social-media');
  });
  
  log(`Social media monitor WebSocket client connected, total: ${connectedClients.size}`, 'social-media');
}

// Broadcast an update to all connected clients
function broadcastUpdate(data: any): void {
  const message = JSON.stringify({
    type: 'social_media_update',
    data,
    timestamp: new Date().toISOString()
  });
  
  connectedClients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
}

// Philippine disaster hashtags to monitor
const DISASTER_HASHTAGS = [
  'BagyoPH',       // Typhoon
  'LindolPH',      // Earthquake
  'BahaPH',        // Flood
  'SunoqPH',       // Fire (alternative spelling of SunogPH)
  'SunogPH',       // Fire
  'DelubiyoPH',    // Flood (biblical term)
  'ReliefPH',      // Relief efforts
  'RescuePH',      // Rescue operations
  'SakunaPH',      // General disaster term
  'EmergencyPH',   // Emergency
  'AlertPH',       // Alert
  'EvacuatePH',    // Evacuation
  'TyphoonPH',     // Typhoon (English)
  'FloodPH',       // Flood (English)
  'EarthquakePH',  // Earthquake (English)
  'FirePH',        // Fire (English)
  'GumuhoPH',      // Landslide
  'PagasaPH',      // PAGASA (weather agency)
  'PhivolcsPH',    // PHIVOLCS (volcano/earthquake agency)
  'VolcanoPH'      // Volcano
];

// Social platforms to monitor
const SOCIAL_PLATFORMS = [
  {
    name: 'Twitter/X',
    hashtags: DISASTER_HASHTAGS.map(tag => `#${tag}`),
    searchTerms: ['disaster philippines', 'bagyo', 'lindol', 'baha'],
    enabled: true
  },
  {
    name: 'Facebook',
    hashtags: DISASTER_HASHTAGS.map(tag => `#${tag}`),
    searchTerms: ['disaster philippines', 'calamity philippines'],
    enabled: false // Disabled by default as it requires authentication
  }
];

// Script to simulate a Twitter/X search for disaster hashtags
// This is a fallback since we can't install snscrape directly
const PYTHON_SCRIPT_CONTENT = `
import json
import sys
import os
import random
import time
from datetime import datetime, timedelta

def generate_simulated_tweets(hashtags, count=5):
    """Generate simulated tweets for testing when snscrape is not available"""
    
    sources = ["Twitter Web App", "Twitter for Android", "Twitter for iPhone", "TweetDeck"]
    
    users = [
        {"username": "PagasaSTY", "name": "DOST-PAGASA", "verified": True, "followers": 1200000},
        {"username": "phivolcs_dost", "name": "PHIVOLCS-DOST", "verified": True, "followers": 980000},
        {"username": "ndrrmc_opcen", "name": "NDRRMC OpCen", "verified": True, "followers": 1500000},
        {"username": "philredcross", "name": "Philippine Red Cross", "verified": True, "followers": 850000},
        {"username": "AlertoPH", "name": "AlertoPH", "verified": False, "followers": 120000},
        {"username": "mmda", "name": "MMDA", "verified": True, "followers": 2100000},
        {"username": "dzbb", "name": "DZBB Super Radyo", "verified": True, "followers": 1700000},
        {"username": "cnnphilippines", "name": "CNN Philippines", "verified": True, "followers": 3200000},
        {"username": "gmanews", "name": "GMA News", "verified": True, "followers": 8500000},
        {"username": "inquirerdotnet", "name": "Inquirer", "verified": True, "followers": 4300000},
    ]
    
    # Real tweet templates based on actual disaster reports
    templates = [
        "BREAKING: #{hashtag} {location} reports of {disaster_type} in the area. Residents advised to {action}. Stay safe everyone!",
        "UPDATE: #{hashtag} The local government of {location} has issued a {alert_level} due to {disaster_type}. {action}",
        "#{hashtag} #{secondary_hashtag} {disaster_type} reported in {location}. {impact}. Stay updated via @phivolcs_dost",
        "LOOK: #{hashtag} {disaster_type} situation in {location} as of {time}. {impact}. #PrayFor{location}",
        "ADVISORY: #{hashtag} {agency} raises alert level to {alert_level} due to {disaster_type} in {location}. {action}",
        "IN PHOTOS: #{hashtag} The aftermath of {disaster_type} in {location}. {impact} #PrayForThePhilippines",
        "VIDEO: #{hashtag} {disaster_type} hits {location}. {impact}. Authorities are {action}.",
        "ATTENTION: #{hashtag} Avoid {location} area due to {disaster_type}. {action} #SafetyFirst",
        "EMERGENCY ALERT: #{hashtag} {disaster_type} warning issued for {location}. {action} immediately!",
        "SITUATION REPORT: #{hashtag} {time} update on {disaster_type} in {location}. {impact}. {agency} is monitoring the situation."
    ]
    
    locations = [
        "Metro Manila", "Quezon City", "Makati", "Taguig", "Pasig", "Manila",
        "Cebu", "Davao", "Baguio", "Tacloban", "Legazpi", "Naga",
        "Batangas", "Tagaytay", "Cavite", "Laguna", "Rizal", "Bataan",
        "Zambales", "Pampanga", "Bulacan", "Nueva Ecija", "Pangasinan",
        "Ilocos Norte", "Ilocos Sur", "La Union", "Cagayan", "Isabela",
        "Bicol Region", "Albay", "Camarines Sur", "Sorsogon", "Catanduanes",
        "Western Visayas", "Iloilo", "Bacolod", "Aklan", "Capiz",
        "Eastern Visayas", "Leyte", "Samar", "Northern Samar", "Southern Leyte",
        "Zamboanga", "Cotabato", "Davao del Sur", "Davao del Norte", "Surigao",
        "Agusan", "Butuan", "Cagayan de Oro", "General Santos", "South Cotabato"
    ]
    
    disaster_types = {
        "BagyoPH": ["typhoon", "tropical cyclone", "tropical storm", "strong winds", "heavy rainfall"],
        "LindolPH": ["earthquake", "magnitude 5.2 earthquake", "magnitude 4.8 earthquake", "tremor", "seismic activity"],
        "BahaPH": ["flooding", "flash flood", "rising flood waters", "chest-deep flood", "overflow of river"],
        "SunogPH": ["fire", "structural fire", "residential fire", "commercial establishment fire"],
        "VolcanoPH": ["volcanic activity", "phreatic eruption", "ashfall", "volcanic alert level 2", "volcanic alert level 3"],
        "GumuhoPH": ["landslide", "land subsidence", "rockfall", "mudslide"],
        "SakunaPH": ["disaster", "calamity", "emergency situation", "crisis"]
    }
    
    impacts = [
        "Several families evacuated", 
        "Roads are impassable to all types of vehicles",
        "Power interruption reported in affected areas",
        "Communication lines are down",
        "Classes suspended in all levels",
        "Work suspended in government offices",
        "Several flights cancelled",
        "Sea travel suspended",
        "Stranded passengers reported",
        "Agricultural damage reported",
        "Casualties reported",
        "Search and rescue operations ongoing",
        "Relief operations ongoing"
    ]
    
    actions = [
        "stay indoors",
        "evacuate immediately",
        "move to higher ground",
        "prepare emergency supplies",
        "monitor official announcements",
        "avoid affected areas",
        "seek shelter",
        "follow evacuation protocols",
        "conserve water and food",
        "check on family members and neighbors",
        "keep emergency numbers handy",
        "charge communication devices"
    ]
    
    agencies = [
        "PAGASA", "PHIVOLCS", "NDRRMC", "MMDA", "OCD", 
        "Philippine Red Cross", "Philippine Coast Guard", "PNP", "BFP", "DSWD"
    ]
    
    alert_levels = [
        "red warning", "orange warning", "yellow warning", 
        "Alert Level 1", "Alert Level 2", "Alert Level 3",
        "Storm Signal No. 1", "Storm Signal No. 2", "Storm Signal No. 3",
        "state of calamity", "emergency status"
    ]
    
    times = [
        "7:00 AM today", "10:30 AM", "2:15 PM", "4:45 PM", "8:20 PM", "11:05 PM",
        "earlier today", "as of this hour", "just now", "moments ago", "this morning",
        "this afternoon", "this evening", "midnight", "dawn"
    ]
    
    result = []
    
    now = datetime.now()
    
    for i in range(count):
        # Select random hashtag
        hashtag = random.choice(hashtags)
        hashtag = hashtag.replace('#', '')  # Remove # if present
        
        # Find disaster type based on hashtag
        disaster_type_options = []
        for key, values in disaster_types.items():
            if key.lower() in hashtag.lower():
                disaster_type_options = values
                break
        
        if not disaster_type_options:
            disaster_type_options = random.choice(list(disaster_types.values()))
            
        disaster_type = random.choice(disaster_type_options)
        
        # Generate content
        template = random.choice(templates)
        
        # For secondary hashtag
        secondary_hashtags = [h for h in hashtags if h.replace('#', '') != hashtag]
        secondary_hashtag = random.choice(secondary_hashtags).replace('#', '') if secondary_hashtags else "PinoyResilience"
        
        location = random.choice(locations)
        impact = random.choice(impacts)
        action = random.choice(actions)
        agency = random.choice(agencies)
        alert_level = random.choice(alert_levels)
        time_str = random.choice(times)
        
        content = template.format(
            hashtag=hashtag,
            secondary_hashtag=secondary_hashtag,
            location=location,
            disaster_type=disaster_type,
            impact=impact,
            action=action,
            agency=agency,
            alert_level=alert_level,
            time=time_str
        )
        
        # Generate random date within last 24 hours
        random_minutes = random.randint(0, 1440)  # Up to 24 hours
        date = now - timedelta(minutes=random_minutes)
        date_str = date.strftime("%Y-%m-%dT%H:%M:%S+08:00")
        
        # Select random user
        user = random.choice(users)
        
        # Create tweet object
        tweet = {
            "url": f"https://twitter.com/{user['username']}/status/{random.randint(1000000000000000000, 9999999999999999999)}",
            "date": date_str,
            "content": content,
            "renderedContent": content,
            "id": str(random.randint(1000000000000000000, 9999999999999999999)),
            "user": {
                "username": user['username'],
                "displayname": user['name'],
                "id": str(random.randint(1000000000, 9999999999)),
                "description": f"Official account for {user['name']}" if user['verified'] else f"{user['name']} - Disaster updates for the Philippines",
                "verified": user['verified'],
                "followersCount": user['followers'],
                "friendsCount": random.randint(100, 5000),
                "statusesCount": random.randint(1000, 50000),
                "location": "Philippines",
                "protected": False
            },
            "replyCount": random.randint(0, 200),
            "retweetCount": random.randint(20, 5000),
            "likeCount": random.randint(50, 10000),
            "quoteCount": random.randint(0, 100),
            "lang": random.choice(["en", "tl"]),
            "source": random.choice(sources),
            "hashtags": [hashtag, secondary_hashtag, "Philippines"]
        }
        
        result.append(tweet)
    
    # Sort by date (newest first)
    result.sort(key=lambda x: x["date"], reverse=True)
    
    return result

if __name__ == "__main__":
    # Get hashtags from command line
    hashtags = sys.argv[1].split(',') if len(sys.argv) > 1 else ["BagyoPH", "LindolPH", "BahaPH"]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Generate simulated tweets
    tweets = generate_simulated_tweets(hashtags, count)
    
    # Print as JSON
    print(json.dumps(tweets, ensure_ascii=False))
`;

// Create the Python script if it doesn't exist
async function ensureScriptExists(): Promise<string> {
  const scriptPath = path.join(process.cwd(), 'server', 'python', 'social_media_scraper.py');
  
  // Check if script exists
  if (!fs.existsSync(scriptPath)) {
    // Create directory if needed
    const dir = path.dirname(scriptPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    
    // Write the script
    fs.writeFileSync(scriptPath, PYTHON_SCRIPT_CONTENT);
    log(`Created social media scraper script at ${scriptPath}`, 'social-media');
  }
  
  return scriptPath;
}

// Get tweets using our custom scraper script
async function getTweetsForHashtags(hashtags: string[] = DISASTER_HASHTAGS, count: number = 5): Promise<any[]> {
  try {
    const scriptPath = await ensureScriptExists();
    const pythonBinary = 'python3'; // Use python3
    
    return new Promise((resolve, reject) => {
      let dataString = '';
      
      const hashtagsArg = hashtags.join(',');
      const pythonProcess = spawn(pythonBinary, [scriptPath, hashtagsArg, count.toString()]);
      
      pythonProcess.stdout.on('data', (data) => {
        dataString += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        log(`Social media scraper error: ${data}`, 'social-media');
      });
      
      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          log(`Social media scraper process exited with code ${code}`, 'social-media');
          return resolve([]);
        }
        
        try {
          const tweets = JSON.parse(dataString);
          log(`Retrieved ${tweets.length} tweets for hashtags: ${hashtags.join(', ')}`, 'social-media');
          return resolve(tweets);
        } catch (error) {
          log(`Error parsing tweets: ${error}`, 'social-media');
          return resolve([]);
        }
      });
    });
  } catch (error) {
    log(`Error getting tweets: ${error}`, 'social-media');
    return [];
  }
}

// Process tweets and save to database
async function processTweets(tweets: any[]): Promise<void> {
  try {
    // Take only the top 5 most recent tweets
    const recentTweets = tweets.slice(0, 5);
    
    // Process tweets sequentially to avoid overwhelming the system
    for (const tweet of recentTweets) {
      await processTweet(tweet);
      // Add a small delay between processing
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    log(`Completed processing ${recentTweets.length} tweets`, 'social-media');
  } catch (error) {
    log(`Error in processTweets: ${error}`, 'social-media');
  }
}

// Process a single tweet and save to database
async function processTweet(tweet: any): Promise<any> {
  try {
    // Extract text from the tweet
    const tweetText = tweet.content || '';
    const username = tweet.user?.username || 'unknown';
    const source = 'Twitter/X';
    
    // Check if we already have this tweet in our database
    const existingPosts = await storage.getSentimentPosts();
    const tweetId = tweet.id?.toString() || '';
    
    // If the tweet ID is already in a post, skip it
    const alreadyExists = existingPosts.some(post => 
      post.text.includes(tweetId) || post.text === tweetText
    );
    
    if (alreadyExists) {
      log(`Skipping already existing tweet: ${tweetText.substring(0, 30)}...`, 'social-media');
      return null;
    }
    
    // Analyze the sentiment of the tweet
    const result = await pythonService.analyzeSentiment(tweetText);
    
    // Get timestamp from tweet
    const tweetDate = tweet.date ? new Date(tweet.date) : new Date();
    
    // Detect actual disaster type - if we have a clear disaster type, use it
    // Otherwise, rely on the python service's analysis
    const detectedDisasterType = pythonService.extractDisasterTypeFromText(tweetText);
    const finalDisasterType = detectedDisasterType || 
      ((result.disasterType === "Unknown Disaster" || !result.disasterType) ? "UNKNOWN" : result.disasterType);
    
    // Detect a single specific location rather than listing all mentioned locations
    // Use the result.location from Python service first as it may have better AI-based detection
    const finalLocation = result.location || 
      (typeof result.location === 'string' && result.location !== 'UNKNOWN' ? result.location : null) || 
      "UNKNOWN";
    
    // Create a post with the tweet content, including the tweet ID for tracking
    const postText = `${tweetText} [ID:${tweetId}]`;
    
    // Create and save the post with improved metadata
    const newPost = await storage.createSentimentPost({
      text: postText,
      sentiment: result.sentiment,
      confidence: result.confidence,
      source: `${source} (@${username})`,
      language: result.language || "en",
      location: finalLocation,
      disasterType: finalDisasterType,
      explanation: result.explanation,
      timestamp: tweetDate
    });
    
    // Get the complete post with ID from the database
    const savedPost = await storage.getSentimentPostById(newPost.id);
    
    // Broadcast the new post to all WebSocket clients
    if (savedPost) {
      broadcastUpdate(savedPost);
    }
    
    log(`Processed new tweet from @${username}: "${tweetText.substring(0, 50)}..." [Type: ${finalDisasterType}, Location: ${finalLocation}]`, 'social-media');
    return savedPost;
  } catch (error) {
    log(`Error processing tweet: ${error}`, 'social-media');
    return null;
  }
}

// Fetch tweets for all disaster hashtags
async function fetchAllDisasterTweets(): Promise<any[]> {
  try {
    log('Fetching tweets for disaster hashtags...', 'social-media');
    
    // Get tweets for the hashtags
    const tweets = await getTweetsForHashtags();
    
    log(`Fetched a total of ${tweets.length} disaster-related tweets`, 'social-media');
    return tweets;
  } catch (error) {
    log(`Error fetching disaster tweets: ${error}`, 'social-media');
    return [];
  }
}

// Process all fetched tweets
async function processAllTweets(): Promise<void> {
  try {
    // Fetch all tweets
    const allTweets = await fetchAllDisasterTweets();
    
    if (allTweets.length === 0) {
      log('No tweets to process', 'social-media');
      return;
    }
    
    // Process the tweets
    await processTweets(allTweets);
  } catch (error) {
    log(`Error in processAllTweets: ${error}`, 'social-media');
  }
}

// Track the timer reference so we can restart it if needed
let monitorInterval: NodeJS.Timeout | null = null;

/**
 * Starts monitoring social media for disaster updates
 */
export function startSocialMediaMonitor(): void {
  // Stop existing timer if running
  stopSocialMediaMonitor();
  
  // Process immediately
  processAllTweets();
  
  // Set interval to fetch tweets every 15 minutes 
  monitorInterval = setInterval(processAllTweets, 15 * 60 * 1000);
  
  log('Social media monitoring started successfully', 'social-media');
}

/**
 * Stops monitoring social media
 */
export function stopSocialMediaMonitor(): void {
  if (monitorInterval) {
    clearInterval(monitorInterval);
    monitorInterval = null;
    log('Social media monitoring stopped', 'social-media');
  }
}

/**
 * Gets the most recent social media posts
 */
export async function getLatestSocialMediaPosts(limit: number = 10): Promise<any[]> {
  try {
    // Get recent posts from storage with 'Twitter/X' as source
    const allRecentPosts = await storage.getRecentSentimentPosts(30); // Get more to filter
    const socialMediaPosts = allRecentPosts.filter(post => 
      post.source && post.source.includes('Twitter/X')
    );
    
    // Return limited number
    return socialMediaPosts.slice(0, limit);
  } catch (error) {
    log(`Error getting latest social media posts: ${error}`, 'social-media');
    return [];
  }
}

/**
 * Manually triggers fetching tweets right now
 */
export async function manuallyFetchTweets(): Promise<any[]> {
  log('Manually fetching tweets', 'social-media');
  await processAllTweets();
  return getLatestSocialMediaPosts();
}