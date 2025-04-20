/**
 * Automatic Post Generator for PanicSense App
 * Simulates social media posts about disasters without requiring external APIs
 */

import { storage } from '../storage';
import { pythonService } from '../python-service';
import { WebSocket } from 'ws';
import { log } from '../vite';

// Track the timer reference so we can restart it if needed
let postGeneratorInterval: NodeJS.Timeout | null = null;

// Connected WebSocket clients that will receive real-time updates
const connectedClients = new Set<WebSocket>();

// List of realistic disaster-related posts (without sensitive/private information)
const disasterRelatedPosts = [
  // Typhoon-related posts
  "Malakas ang hangin dito sa amin, may paparating na bagyo #BagyoPH",
  "Flash flood warning sa Eastern Samar. Mag-ingat po tayong lahat!",
  "Nag-umpisa na ang pag-ulan dito sa Manila, hindi pa naman masyadong malakas",
  "Road closures reported along Kennon Road due to landslides from heavy rain",
  "Government has issued evacuation orders for coastal areas. #typhoon",
  
  // Earthquake-related posts
  "May naramdaman ba kayong lindol? Nagising ako bigla",
  "Magnitude 4.2 earthquake just occurred in Batangas, no damage reported yet",
  "Tremors felt in Quezon City! Everyone ok? #lindolPH",
  "Our office building in Makati evacuated after earthquake, waiting for safety clearance",
  "Nasiraan ng kuryente after yung earthquake kanina, sino pa po walang power?",
  
  // Flood-related posts
  "Lubog na ang Marikina River, first alarm na. Heads up sa mga kababayan natin doon",
  "Naputol na ang daan sa Rizal Avenue dahil sa baha, hanap kayo ibang route",
  "Relief goods needed for families affected by flooding in Cagayan Valley",
  "Water level rising quickly in our area. Local officials are monitoring the situation",
  "Classes suspended tomorrow in all levels due to continuous rainfall and flooding",
  
  // Volcanic activity
  "Ash fall warning in Batangas and Cavite due to increased activity of Taal Volcano",
  "Alert level 3 raised at Mayon Volcano, precautionary evacuations underway",
  "PHIVOLCS issues advisory on Kanlaon Volcano after series of earthquakes detected",
  "May nakikita kaming usok mula sa bulkan, normal lang ba to?",
  "Fine ash has begun falling in our area near Taal, everyone indoors please",
  
  // Fire incidents
  "Big fire reported in Quiapo area, fire trucks on the way",
  "May nasusunog na warehouse sa industrial park, evacuating nearby buildings",
  "Fire in residential area in Pasig, please avoid Commonwealth Avenue for now",
  "Bureau of Fire Protection responding to multiple fire alarms due to lightning strikes",
  "Sunog sa palengke! Mabilis kumalat dahil maraming materials na flamable!",
  
  // General emergency/disaster posts
  "Red alert issued for several regions in Mindanao due to severe weather system",
  "Roads toward Baguio City temporarily closed due to landslides",
  "Rescue operations ongoing for stranded residents in Marikina",
  "NDRRMC advises residents of coastal towns to prepare emergency kits",
  "Power outage affecting multiple barangays in Metro Manila after the storm"
];

// Variations to add realism to the posts
const postVariations = [
  "Anyone experiencing the same in {location}?",
  "Please share updates if you're in the affected area!",
  "Stay safe everyone!",
  "#EmergencyAlert",
  "Authorities are responding to the situation.",
  "According to PAGASA...",
  "BREAKING NEWS:",
  "Update:",
  "Confirmed:",
  "Developing situation:",
  "Need help! 🆘",
  "Please RT for awareness"
];

// List of Philippine locations for realistic post generation
const locations = [
  "Manila", "Quezon City", "Davao", "Cebu", "Makati", 
  "Taguig", "Pasig", "Cagayan", "Bicol", "Samar", 
  "Leyte", "Tacloban", "Batanes", "Mindanao", "Luzon", 
  "Visayas", "Palawan", "Mindoro", "Batangas", "Cavite", 
  "Laguna", "Albay", "Baguio", "Zambales", "Pampanga", 
  "Bulacan", "Iloilo", "Bacolod", "Zamboanga"
];

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
    
    log(`Broadcasted new post to ${connectedClients.size} clients`, 'auto-posts');
  } catch (error) {
    log(`Error broadcasting new post: ${error}`, 'auto-posts');
  }
}

/**
 * Adds a WebSocket client to the list of connected clients
 */
export function addWebSocketClient(ws: WebSocket) {
  connectedClients.add(ws);
  
  // Remove the client when it disconnects
  ws.on('close', () => {
    connectedClients.delete(ws);
    log(`Real-time feed client disconnected, ${connectedClients.size} remaining`, 'auto-posts');
  });
  
  log(`New real-time feed client connected, total clients: ${connectedClients.size}`, 'auto-posts');
}

/**
 * Generates a single new disaster-related post
 */
async function generatePost() {
  try {
    // Pick a random post base
    const basePost = disasterRelatedPosts[Math.floor(Math.random() * disasterRelatedPosts.length)];
    
    // Sometimes add a variation for realism
    let postText = basePost;
    if (Math.random() > 0.7) {
      const variation = postVariations[Math.floor(Math.random() * postVariations.length)];
      // Replace {location} placeholder if present
      if (variation.includes('{location}')) {
        const location = locations[Math.floor(Math.random() * locations.length)];
        postText = `${basePost} ${variation.replace('{location}', location)}`;
      } else {
        postText = `${basePost} ${variation}`;
      }
    }
    
    // Analyze the sentiment of the post using the existing system
    const result = await pythonService.analyzeSentiment(postText);
    
    // Create and save the post
    const newPost = await storage.createSentimentPost({
      text: postText,
      sentiment: result.sentiment,
      confidence: result.confidence,
      source: "Real-Time Feed",
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
    
    log(`Generated new auto-post: "${postText.substring(0, 50)}..." with sentiment: ${result.sentiment}`, 'auto-posts');
    return savedPost;
  } catch (error) {
    log(`Error generating post: ${error}`, 'auto-posts');
    return null;
  }
}

/**
 * Randomly generates a batch of 1-3 new posts
 */
async function generatePostBatch() {
  try {
    const count = Math.floor(Math.random() * 3) + 1; // 1-3 posts at a time
    log(`Generating batch of ${count} new posts`, 'auto-posts');
    
    for (let i = 0; i < count; i++) {
      await generatePost();
      // Add small delay between posts
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  } catch (error) {
    log(`Error generating post batch: ${error}`, 'auto-posts');
  }
}

/**
 * Gets the most recent posts for the real-time feed (15-20 posts)
 */
export async function getLatestPosts(limit: number = 20) {
  try {
    return await storage.getRecentSentimentPosts(limit);
  } catch (error) {
    log(`Error getting latest posts: ${error}`, 'auto-posts');
    return [];
  }
}

/**
 * Starts the automatic post generator
 * Will run every 10-20 minutes to ensure we get 15-20 posts per hour
 */
export function startAutoPostGenerator() {
  // Stop existing timer if running
  stopAutoPostGenerator();
  
  // Generate the first batch immediately
  generatePostBatch();
  
  // Set interval for future batches (10-20 minutes)
  // This ensures approximately 15-20 posts per hour
  postGeneratorInterval = setInterval(async () => {
    // Get the interval between 10-20 minutes (in milliseconds)
    const intervalMinutes = Math.floor(Math.random() * 10) + 10;
    const intervalMs = intervalMinutes * 60 * 1000;
    
    // Clear the current interval
    if (postGeneratorInterval) {
      clearInterval(postGeneratorInterval);
    }
    
    // Generate posts now
    await generatePostBatch();
    
    // Set a new random interval
    postGeneratorInterval = setInterval(generatePostBatch, intervalMs);
    
    log(`Next batch of posts scheduled in ${intervalMinutes} minutes`, 'auto-posts');
  }, 15 * 60 * 1000); // Initial interval of 15 minutes
  
  log('Automatic post generator started successfully', 'auto-posts');
}

/**
 * Stops the automatic post generator
 */
export function stopAutoPostGenerator() {
  if (postGeneratorInterval) {
    clearInterval(postGeneratorInterval);
    postGeneratorInterval = null;
    log('Automatic post generator stopped', 'auto-posts');
  }
}

/**
 * Manually triggers the generation of a batch of posts
 * Useful for testing or when you want immediate new content
 */
export async function manuallyGeneratePosts(count: number = 3) {
  log(`Manually generating ${count} posts`, 'auto-posts');
  
  const newPosts = [];
  for (let i = 0; i < count; i++) {
    const post = await generatePost();
    if (post) {
      newPosts.push(post);
    }
    // Small delay between posts
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  
  return newPosts;
}