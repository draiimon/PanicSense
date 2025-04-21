/**
 * TEXT ANALYSIS FIX
 * Direct implementation for real-time text analysis on Render
 */

import { pool } from './server/db.js';
import fs from 'fs';
import path from 'path';
import express from 'express';

// Create simplified router for text analysis
const router = express.Router();

// Simplified sentiment analysis function (server-side only)
function analyzeSentiment(text) {
  console.log(`Analyzing text: "${text.substring(0, 50)}..."`);
  
  // Simple disaster-related keywords
  const disasterKeywords = [
    'typhoon', 'bagyo', 'flood', 'baha', 'earthquake', 'lindol', 'landslide', 'evacuation',
    'warning', 'alert', 'disaster', 'emergency', 'rescue', 'damage', 'casualties',
    'storm', 'heavy rain', 'fire', 'sunog', 'tsunami', 'victims', 'trapped', 'stranded'
  ];
  
  // Location keywords
  const locationKeywords = [
    'Manila', 'Quezon', 'Cebu', 'Davao', 'Visayas', 'Mindanao', 'Luzon', 'Bicol',
    'Cagayan', 'Benguet', 'Marikina', 'Rizal', 'Batangas', 'Laguna', 'Cavite',
    'Bataan', 'Zambales', 'Pampanga', 'Philippines', 'Pilipinas'
  ];
  
  // Check for disaster keywords
  const textLower = text.toLowerCase();
  const foundDisasterKeywords = disasterKeywords.filter(keyword => 
    textLower.includes(keyword.toLowerCase())
  );
  
  // Check for location keywords
  const foundLocationKeywords = locationKeywords.filter(keyword => 
    new RegExp(`\\b${keyword}\\b`, 'i').test(text)
  );
  
  // Determine if disaster-related
  const isDisasterRelated = foundDisasterKeywords.length > 0;
  
  // Disaster type detection
  let disasterType = 'Not Specified';
  if (textLower.includes('typhoon') || textLower.includes('bagyo') || textLower.includes('storm')) {
    disasterType = 'Typhoon';
  } else if (textLower.includes('flood') || textLower.includes('baha')) {
    disasterType = 'Flood';
  } else if (textLower.includes('earthquake') || textLower.includes('lindol')) {
    disasterType = 'Earthquake';
  } else if (textLower.includes('fire') || textLower.includes('sunog')) {
    disasterType = 'Fire';
  } else if (textLower.includes('landslide')) {
    disasterType = 'Landslide';
  } else if (foundDisasterKeywords.length > 0) {
    disasterType = 'Other Disaster';
  }
  
  // Sentiment detection (simplified)
  let sentiment = 'Neutral';
  let confidence = 0.75;
  
  // Negative words
  const negativeWords = ['damage', 'destroyed', 'dead', 'casualties', 'victims', 'trapped', 'stranded', 
    'emergency', 'danger', 'panic', 'fear', 'terrible', 'scary', 'death', 'dying'];
  
  // Positive words  
  const positiveWords = ['rescue', 'saved', 'recovery', 'help', 'assist', 'support', 'relief', 
    'evacuated', 'safe', 'survived', 'donations', 'volunteers'];
  
  // Warning words
  const warningWords = ['warning', 'alert', 'evacuate', 'caution', 'danger', 'prepare', 'approaching'];
  
  // Count occurrences
  const negativeCount = negativeWords.filter(word => textLower.includes(word)).length;
  const positiveCount = positiveWords.filter(word => textLower.includes(word)).length;
  const warningCount = warningWords.filter(word => textLower.includes(word)).length;
  
  // Determine sentiment
  if (warningCount > 0 && warningCount >= negativeCount && warningCount >= positiveCount) {
    sentiment = 'Warning';
    confidence = 0.80 + (warningCount * 0.03);
  } else if (negativeCount > positiveCount) {
    sentiment = 'Concern';
    confidence = 0.70 + (negativeCount * 0.04);
  } else if (positiveCount > negativeCount) {
    sentiment = 'Support';
    confidence = 0.75 + (positiveCount * 0.03);
  }
  
  // Determine location
  let location = foundLocationKeywords.length > 0 ? 
    foundLocationKeywords.join(', ') + ', Philippines' : 
    'Philippines';
  
  // Caping confidence at 0.95
  confidence = Math.min(confidence, 0.95);
  
  return {
    sentiment,
    confidence,
    language: /[ñÑáéíóúÁÉÍÓÚ]/.test(text) ? 'tl' : 'en',
    disasterType: isDisasterRelated ? disasterType : 'Not Disaster-Related',
    location,
    explanation: `Text analysis found ${foundDisasterKeywords.length} disaster keywords and ${foundLocationKeywords.length} location references.`,
    isDisasterRelated
  };
}

// Direct real-time text analysis endpoint 
router.post('/api/direct-analyze-text', async (req, res) => {
  try {
    const { text } = req.body;
    
    if (!text) {
      return res.status(400).json({ error: "Text is required" });
    }
    
    console.log(`Received text analysis request for: "${text.substring(0, 30)}..."`);
    
    // Perform analysis
    const result = analyzeSentiment(text);
    
    // Determine if we should save this
    const shouldSave = result.isDisasterRelated;
    
    let savedPost = null;
    
    if (shouldSave) {
      try {
        // Save to database
        const query = `
          INSERT INTO sentiment_posts 
            (text, source, language, sentiment, confidence, disaster_type, location, explanation, processed_by, timestamp)
          VALUES 
            ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
          RETURNING *
        `;
        
        const params = [
          text,
          'Direct Text Input',
          result.language,
          result.sentiment,
          result.confidence,
          result.disasterType,
          result.location,
          result.explanation,
          'Real-time API',
          new Date()
        ];
        
        const insertResult = await pool.query(query, params);
        savedPost = insertResult.rows[0];
        console.log(`Saved sentiment post with ID: ${savedPost.id}`);
        
        // Check if we need to create a disaster event
        // Only create event if confidence is high and explicitly disaster related
        if (result.confidence > 0.8 && result.disasterType !== 'Not Disaster-Related' && result.disasterType !== 'Not Specified') {
          // Check if similar event exists in last 24 hours
          const existingEvents = await pool.query(`
            SELECT * FROM disaster_events 
            WHERE event_type = $1 
            AND location LIKE $2
            AND timestamp > NOW() - INTERVAL '24 hours'
            LIMIT 1
          `, [result.disasterType, `%${result.location.split(',')[0]}%`]);
          
          if (existingEvents.rows.length > 0) {
            // Update existing event
            const event = existingEvents.rows[0];
            await pool.query(`
              UPDATE disaster_events
              SET source_count = source_count + 1,
                  timestamp = NOW()
              WHERE id = $1
            `, [event.id]);
            
            console.log(`Updated existing disaster event: ${event.id}`);
          } else {
            // Create new event
            const eventName = `${result.disasterType} Alert`;
            const description = `Based on real-time analysis of user-submitted text. Location: ${result.location}`;
            
            await pool.query(`
              INSERT INTO disaster_events
                (name, description, location, severity, event_type, source_count, timestamp)
              VALUES
                ($1, $2, $3, $4, $5, $6, $7)
            `, [
              eventName,
              description,
              result.location,
              'Medium',
              result.disasterType,
              1,
              new Date()
            ]);
            
            console.log(`Created new disaster event: ${eventName}`);
          }
        }
      } catch (dbError) {
        console.error("Database error saving sentiment:", dbError);
        // Continue without saving
      }
    }
    
    // Return the result
    res.json({
      post: savedPost || {
        id: -1,
        text,
        timestamp: new Date().toISOString(),
        source: 'Manual Input (Not Saved)',
        language: result.language,
        sentiment: result.sentiment,
        confidence: result.confidence,
        location: result.location,
        disasterType: result.disasterType,
        explanation: result.explanation
      },
      saved: shouldSave,
      message: shouldSave 
        ? "Disaster-related content detected and saved to database." 
        : "Non-disaster content detected. Analysis shown but not saved to database."
    });
    
  } catch (error) {
    console.error("Error in direct text analysis:", error);
    res.status(500).json({
      error: "Failed to analyze text",
      details: error.message || String(error)
    });
  }
});

// Check if server is already running
let server = null;
if (typeof global.__directTextAnalysisServer === 'undefined') {
  // Create a direct Express server on a different port
  const directApp = express();
  directApp.use(express.json());
  directApp.use(router);
  
  // Store in global to prevent duplicate servers
  try {
    server = directApp.listen(10001, '0.0.0.0', () => {
      console.log("✅ Direct text analysis server started on port 10001");
      
      // Store for future reference
      global.__directTextAnalysisServer = server;
    });
  } catch (error) {
    console.error("Error starting direct text analysis server:", error);
  }
}

// Also export the router so it can be used in main app
export default router;

// Print instructions for use
console.log(`
🔍 REAL-TIME TEXT ANALYSIS FIX

To use the direct text analysis endpoint, send POST requests to:
http://localhost:10001/api/direct-analyze-text

With JSON body:
{"text": "Your text to analyze here"}

You can also use this endpoint through the main app if it's restarted
after importing this module.
`);

// BONUS: Add some test examples
console.log("Testing text analysis functionality...");
const testTexts = [
  "Typhoon approaching Eastern Visayas with strong winds and heavy rain expected.",
  "Flash floods reported in Marikina after continuous rain.",
  "Residents in coastal areas of Quezon Province advised to evacuate due to storm surge risk.",
  "Relief operations ongoing in Albay for families affected by typhoon."
];

for (const text of testTexts) {
  const result = analyzeSentiment(text);
  console.log(`
Test: "${text}"
Result: ${result.sentiment} (${Math.round(result.confidence * 100)}%)
Type: ${result.disasterType}
Location: ${result.location}
`);
}