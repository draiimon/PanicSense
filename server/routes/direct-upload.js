
/**
 * DIRECT UPLOAD ENDPOINT
 * Simple file upload implementation that works without Python dependency
 */

import express from 'express';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { pool } from '../db.js';
import { parse } from 'csv-parse';

// Get directory name in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configure multer storage
const storage = multer.diskStorage({
  destination: function(req, file, cb) {
    const uploadsDir = path.join(__dirname, '..', '..', 'uploads');
    
    // Create uploads directory if it doesn't exist
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
    }
    
    cb(null, uploadsDir);
  },
  filename: function(req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + '-' + file.originalname);
  }
});

const upload = multer({ storage: storage });

const router = express.Router();

// Simple text analysis function
function analyzeText(text) {
  // Simple disaster-related keywords
  const disasterKeywords = [
    'typhoon', 'bagyo', 'flood', 'baha', 'earthquake', 'lindol', 'landslide',
    'warning', 'alert', 'disaster', 'emergency', 'rescue', 'damage', 'casualties',
    'storm', 'heavy rain', 'fire', 'sunog', 'tsunami', 'victims', 'trapped', 'stranded'
  ];
  
  // Check for disaster keywords
  const textLower = text.toLowerCase();
  const foundDisasterKeywords = disasterKeywords.filter(keyword => 
    textLower.includes(keyword.toLowerCase())
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
  
  // Cap confidence at 0.95
  confidence = Math.min(confidence, 0.95);
  
  return {
    sentiment,
    confidence,
    language: /[ñÑáéíóúÁÉÍÓÚ]/.test(text) ? 'tl' : 'en',
    disasterType,
    isDisasterRelated
  };
}

// Detect location from text (simplified)
function detectLocation(text) {
  const locationKeywords = [
    'Manila', 'Quezon', 'Cebu', 'Davao', 'Visayas', 'Mindanao', 'Luzon', 'Bicol',
    'Cagayan', 'Benguet', 'Marikina', 'Rizal', 'Batangas', 'Laguna', 'Cavite',
    'Bataan', 'Zambales', 'Pampanga'
  ];
  
  const foundLocations = locationKeywords.filter(location => 
    new RegExp(`\\b${location}\\b`, 'i').test(text)
  );
  
  return foundLocations.length > 0 ? 
    foundLocations.join(', ') + ', Philippines' : 
    'Philippines';
}

// Upload session tracking
router.get('/api/active-upload-session', async (req, res) => {
  try {
    const { sessionId } = req.query;
    
    if (!sessionId) {
      return res.json({ sessionId: null });
    }
    
    // Get session from database
    const result = await pool.query('SELECT * FROM upload_sessions WHERE session_id = $1', [sessionId]);
    
    if (result.rows.length === 0) {
      return res.json({ sessionId: null });
    }
    
    const session = result.rows[0];
    return res.json(session);
  } catch (error) {
    console.error('Error getting upload session:', error);
    return res.status(500).json({ error: 'Failed to get upload session' });
  }
});

// Direct file upload endpoint
router.post('/api/direct-upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    const file = req.file;
    const sessionId = uuidv4();
    const filename = file.originalname;
    const filePath = file.path;
    
    console.log(`Processing uploaded file: ${filename}`);
    
    // Create upload session
    await pool.query(`
      INSERT INTO upload_sessions (
        session_id, status, filename, file_path, progress
      ) VALUES ($1, $2, $3, $4, $5)
    `, [
      sessionId,
      'processing',
      filename,
      filePath,
      JSON.stringify({ current: 0, total: 0 })
    ]);
    
    // Process file asynchronously
    processFile(filePath, filename, sessionId).then(() => {
      console.log(`File processing completed for session ${sessionId}`);
    }).catch(error => {
      console.error(`Error processing file for session ${sessionId}:`, error);
      updateSession(sessionId, 'error', { error: error.message });
    });
    
    // Return session ID
    res.json({ 
      sessionId,
      message: 'File uploaded and processing started'
    });
  } catch (error) {
    console.error('Error uploading file:', error);
    res.status(500).json({ error: 'Failed to upload file' });
  }
});

// Update upload session status
async function updateSession(sessionId, status, progress) {
  try {
    await pool.query(`
      UPDATE upload_sessions 
      SET status = $1, progress = $2, updated_at = NOW() 
      WHERE session_id = $3
    `, [
      status,
      JSON.stringify(progress),
      sessionId
    ]);
  } catch (error) {
    console.error(`Error updating session ${sessionId}:`, error);
  }
}

// Process uploaded CSV file
async function processFile(filePath, filename, sessionId) {
  console.log(`Starting to process file: ${filename}`);
  
  try {
    // Read file
    const fileContent = fs.readFileSync(filePath, 'utf8');
    
    // Create an analyzed file record
    const fileResult = await pool.query(`
      INSERT INTO analyzed_files (
        filename, status, source_count, created_at, updated_at, timestamp
      ) VALUES ($1, $2, $3, NOW(), NOW(), NOW())
      RETURNING id
    `, [
      filename,
      'processing',
      0
    ]);
    
    const fileId = fileResult.rows[0].id;
    console.log(`Created analyzed file record with ID: ${fileId}`);
    
    // Update session with file ID
    await updateSession(sessionId, 'processing', { 
      current: 0, 
      total: 0,
      fileId
    });
    
    // Parse CSV file
    const records = [];
    
    const parser = fs
      .createReadStream(filePath)
      .pipe(parse({
        columns: true,
        skip_empty_lines: true,
        trim: true
      }));
    
    for await (const record of parser) {
      records.push(record);
    }
    
    console.log(`Parsed ${records.length} records from CSV`);
    
    // Update progress
    await updateSession(sessionId, 'processing', { 
      current: 0, 
      total: records.length,
      fileId
    });
    
    // Process each record
    const processed = [];
    let currentRow = 0;
    
    for (const record of records) {
      currentRow++;
      
      // Update progress every 10 records
      if (currentRow % 10 === 0) {
        await updateSession(sessionId, 'processing', { 
          current: currentRow, 
          total: records.length,
          fileId
        });
      }
      
      // Get text from record (look for common column names)
      const text = record.text || record.content || record.message || 
                  record.tweet || record.post || Object.values(record)[0];
      
      if (!text) continue;
      
      // Analyze text
      const analysis = analyzeText(text);
      
      // Detect location
      const location = detectLocation(text);
      
      // Save to database
      const source = record.source || record.platform || 'CSV Upload';
      
      const result = await pool.query(`
        INSERT INTO sentiment_posts (
          text, source, language, sentiment, confidence, 
          disaster_type, location, file_id, processed_by, timestamp
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
        RETURNING id
      `, [
        text,
        source,
        analysis.language,
        analysis.sentiment,
        analysis.confidence,
        analysis.disasterType,
        location,
        fileId,
        'Direct Upload Processor'
      ]);
      
      processed.push(result.rows[0].id);
    }
    
    console.log(`Processed and saved ${processed.length} records`);
    
    // Update file record
    await pool.query(`
      UPDATE analyzed_files 
      SET status = $1, source_count = $2, updated_at = NOW() 
      WHERE id = $3
    `, [
      'completed',
      processed.length,
      fileId
    ]);
    
    // Create event summaries (find common disaster types)
    const disasterTypeCounts = {};
    const disasterTypeResults = await pool.query(`
      SELECT disaster_type, COUNT(*) as count 
      FROM sentiment_posts 
      WHERE file_id = $1 
      GROUP BY disaster_type
    `, [fileId]);
    
    for (const row of disasterTypeResults.rows) {
      if (row.disaster_type !== 'Not Specified' && row.disaster_type !== 'Not Disaster-Related') {
        disasterTypeCounts[row.disaster_type] = parseInt(row.count);
      }
    }
    
    // Create disaster events for common types
    for (const [disasterType, count] of Object.entries(disasterTypeCounts)) {
      if (count >= 3) { // Only create events if there are at least 3 mentions
        // Find common locations
        const locationResults = await pool.query(`
          SELECT location, COUNT(*) as count 
          FROM sentiment_posts 
          WHERE file_id = $1 AND disaster_type = $2
          GROUP BY location
          ORDER BY count DESC
          LIMIT 1
        `, [fileId, disasterType]);
        
        const location = locationResults.rows.length > 0 ? 
          locationResults.rows[0].location : 'Philippines';
        
        // Create event
        await pool.query(`
          INSERT INTO disaster_events (
            name, description, location, severity, event_type, source_count, timestamp
          ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
        `, [
          `${disasterType} Analysis`,
          `Based on analysis of ${count} mentions in uploaded data.`,
          location,
          count > 10 ? 'High' : 'Medium',
          disasterType,
          count
        ]);
        
        console.log(`Created disaster event for ${disasterType}`);
      }
    }
    
    // Update session to completed
    await updateSession(sessionId, 'completed', { 
      current: records.length, 
      total: records.length,
      fileId,
      processedCount: processed.length
    });
    
    console.log(`File processing completed for session ${sessionId}`);
  } catch (error) {
    console.error('Error processing file:', error);
    await updateSession(sessionId, 'error', { error: error.message });
    throw error;
  }
}

export default router;
