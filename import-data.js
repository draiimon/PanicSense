import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';
import { Pool } from '@neondatabase/serverless';
import { drizzle } from 'drizzle-orm/neon-serverless';
import * as schema from './shared/schema.js';
import ws from 'ws';
import { config } from 'dotenv';

// Load environment variables
config();

// Set up WebSocket for Neon
const neonConfig = { webSocketConstructor: ws };

// Check for DATABASE_URL
if (!process.env.DATABASE_URL) {
  throw new Error('DATABASE_URL environment variable is required');
}

// Create database connection
const pool = new Pool({ connectionString: process.env.DATABASE_URL });
const db = drizzle({ client: pool, schema });

async function importData() {
  try {
    console.log('Starting data import...');
    
    // Get all CSV files in the root directory
    const csvFiles = fs.readdirSync(process.cwd()).filter(file => file.endsWith('.csv'));
    console.log(`Found ${csvFiles.length} CSV files to process`);
    
    let totalImportedRecords = 0;
    
    // Import test-data-variant.csv
    const variantDataPath = path.join(process.cwd(), 'test-data-variant.csv');
    if (fs.existsSync(variantDataPath)) {
      console.log(`Reading from ${variantDataPath}`);
      const fileContent = fs.readFileSync(variantDataPath, 'utf8');
      const records = parse(fileContent, {
        columns: true,
        skip_empty_lines: true
      });
      
      console.log(`Found ${records.length} records in test-data-variant.csv`);
      
      // Process and insert records
      for (const record of records) {
        const sentimentPost = {
          text: record.message,
          timestamp: new Date(record.date),
          source: record.platform || 'Unknown',
          language: detectLanguage(record.message),
          sentiment: record.sentiment || 'Neutral',
          confidence: 0.85,
          location: record.place,
          disasterType: record.event_type,
        };
        
        await db.insert(schema.sentimentPosts).values(sentimentPost);
      }
      
      console.log(`Imported ${records.length} records from test-data-variant.csv`);
      totalImportedRecords += records.length;
    } else {
      console.log('File test-data-variant.csv does not exist');
    }
    
    // Import test-data-with-location.csv
    const locationDataPath = path.join(process.cwd(), 'test-data-with-location.csv');
    if (fs.existsSync(locationDataPath)) {
      console.log(`Reading from ${locationDataPath}`);
      const fileContent = fs.readFileSync(locationDataPath, 'utf8');
      const records = parse(fileContent, {
        columns: true,
        skip_empty_lines: true
      });
      
      console.log(`Found ${records.length} records in test-data-with-location.csv`);
      
      // Process and insert records
      for (const record of records) {
        const sentimentPost = {
          text: record.Text,
          timestamp: new Date(record.Timestamp),
          source: record.Source || 'Unknown',
          language: detectLanguage(record.Text),
          sentiment: determineSentiment(record.Text),
          confidence: 0.8,
          location: record.Location,
          disasterType: record.Disaster,
        };
        
        await db.insert(schema.sentimentPosts).values(sentimentPost);
      }
      
      console.log(`Imported ${records.length} records from test-data-with-location.csv`);
      totalImportedRecords += records.length;
    } else {
      console.log('File test-data-with-location.csv does not exist');
    }
    
    // Import test-data.csv
    const testDataPath = path.join(process.cwd(), 'test-data.csv');
    if (fs.existsSync(testDataPath)) {
      console.log(`Reading from ${testDataPath}`);
      const fileContent = fs.readFileSync(testDataPath, 'utf8');
      const records = parse(fileContent, {
        columns: true,
        skip_empty_lines: true
      });
      
      console.log(`Found ${records.length} records in test-data.csv`);
      
      // Process and insert records
      for (const record of records) {
        // Handle different CSV formats
        const text = record.message || record.text || record.Text || record.content || '';
        const timestamp = record.date || record.timestamp || record.Timestamp || new Date();
        const source = record.platform || record.source || record.Source || 'Unknown';
        const location = record.place || record.location || record.Location || null;
        const disasterType = record.event_type || record.disasterType || record.Disaster || record.disaster_type || null;
        const sentiment = record.sentiment || record.Sentiment || determineSentiment(text);
        
        const sentimentPost = {
          text,
          timestamp: new Date(timestamp),
          source,
          language: detectLanguage(text),
          sentiment,
          confidence: 0.82,
          location,
          disasterType,
        };
        
        await db.insert(schema.sentimentPosts).values(sentimentPost);
      }
      
      console.log(`Imported ${records.length} records from test-data.csv`);
      totalImportedRecords += records.length;
    } else {
      console.log('File test-data.csv does not exist');
    }
    
    // Import sample-sentiments.csv
    const sampleDataPath = path.join(process.cwd(), 'sample-sentiments.csv');
    if (fs.existsSync(sampleDataPath)) {
      console.log(`Reading from ${sampleDataPath}`);
      const fileContent = fs.readFileSync(sampleDataPath, 'utf8');
      const records = parse(fileContent, {
        columns: true,
        skip_empty_lines: true
      });
      
      console.log(`Found ${records.length} records in sample-sentiments.csv`);
      
      // Process and insert records
      for (const record of records) {
        // Handle different CSV formats
        const text = record.message || record.text || record.Text || record.content || '';
        const timestamp = record.date || record.timestamp || record.Timestamp || new Date();
        const source = record.platform || record.source || record.Source || 'Social Media';
        const location = record.place || record.location || record.Location || null;
        const disasterType = record.event_type || record.disasterType || record.Disaster || record.disaster_type || null;
        const sentiment = record.sentiment || record.Sentiment || determineSentiment(text);
        
        const sentimentPost = {
          text,
          timestamp: new Date(timestamp),
          source,
          language: detectLanguage(text),
          sentiment,
          confidence: 0.82,
          location,
          disasterType,
        };
        
        await db.insert(schema.sentimentPosts).values(sentimentPost);
      }
      
      console.log(`Imported ${records.length} records from sample-sentiments.csv`);
      totalImportedRecords += records.length;
    } else {
      console.log('File sample-sentiments.csv does not exist');
    }
    
    // Import any other CSV files found
    for (const csvFile of csvFiles) {
      // Skip files we've already processed
      if (['test-data-variant.csv', 'test-data-with-location.csv', 'test-data.csv', 'sample-sentiments.csv'].includes(csvFile)) {
        continue;
      }
      
      console.log(`Reading from ${csvFile}`);
      const filePath = path.join(process.cwd(), csvFile);
      const fileContent = fs.readFileSync(filePath, 'utf8');
      
      try {
        const records = parse(fileContent, {
          columns: true,
          skip_empty_lines: true
        });
        
        console.log(`Found ${records.length} records in ${csvFile}`);
        
        // Process and insert records
        for (const record of records) {
          // Try to identify the relevant fields
          const text = record.message || record.text || record.Text || record.content || '';
          
          // Only process record if we have text content
          if (text) {
            const timestamp = record.date || record.timestamp || record.Timestamp || new Date();
            const source = record.platform || record.source || record.Source || 'Unknown';
            const location = record.place || record.location || record.Location || null;
            const disasterType = record.event_type || record.disasterType || record.Disaster || record.disaster_type || null;
            const sentiment = record.sentiment || record.Sentiment || determineSentiment(text);
            
            const sentimentPost = {
              text,
              timestamp: new Date(timestamp),
              source,
              language: detectLanguage(text),
              sentiment,
              confidence: 0.8,
              location,
              disasterType,
            };
            
            await db.insert(schema.sentimentPosts).values(sentimentPost);
          }
        }
        
        console.log(`Imported ${records.length} records from ${csvFile}`);
        totalImportedRecords += records.length;
      } catch (error) {
        console.error(`Error processing ${csvFile}:`, error.message);
      }
    }
    
    console.log(`Total imported records: ${totalImportedRecords}`);
    
    // Create some disaster events based on the imported data
    const uniqueDisasters = new Set();
    const posts = await db.select().from(schema.sentimentPosts);
    
    posts.forEach(post => {
      if (post.disasterType && !uniqueDisasters.has(post.disasterType)) {
        uniqueDisasters.add(post.disasterType);
      }
    });
    
    console.log(`Found ${uniqueDisasters.size} unique disaster types`);
    
    for (const disasterType of uniqueDisasters) {
      const relatedPosts = posts.filter(post => post.disasterType === disasterType);
      if (relatedPosts.length > 0) {
        const event = {
          name: `${disasterType} Event`,
          description: `A ${disasterType.toLowerCase()} event with ${relatedPosts.length} related posts`,
          timestamp: new Date(Math.min(...relatedPosts.map(p => new Date(p.timestamp).getTime()))),
          location: relatedPosts[0].location,
          type: disasterType,
          sentimentImpact: getMostCommonSentiment(relatedPosts)
        };
        
        await db.insert(schema.disasterEvents).values(event);
      }
    }
    
    console.log('Data import completed successfully!');
    
  } catch (error) {
    console.error('Error during data import:', error);
  } finally {
    process.exit(0);
  }
}

// Helper function to detect language (simple version)
function detectLanguage(text) {
  // Simple detection based on common Filipino words
  const filipinoWords = ['ang', 'ng', 'nang', 'mga', 'ko', 'ako', 'ikaw', 'siya', 'kami', 'tayo', 'kayo', 'sila', 'ito', 'iyan', 'iyon', 'dito', 'diyan', 'doon', 'na', 'pa', 'pala', 'po', 'hindi', 'oo', 'sobrang', 'talaga', 'lamang', 'lang', 'ay', 'at', 'kung', 'kapag', 'dahil', 'dahilan', 'kasi', 'upang', 'para', 'araw', 'gabi', 'umaga', 'tanghali', 'hapon', 'gabi', 'kahapon', 'ngayon', 'bukas'];
  
  const textLower = text.toLowerCase();
  for (const word of filipinoWords) {
    if (textLower.includes(' ' + word + ' ') || 
        textLower.startsWith(word + ' ') || 
        textLower.endsWith(' ' + word) || 
        textLower === word) {
      return 'tl'; // Tagalog language code
    }
  }
  
  return 'en'; // Default to English
}

// Helper function to determine sentiment from text
function determineSentiment(text) {
  const textLower = text.toLowerCase();
  
  const fearWords = ['fear', 'afraid', 'scared', 'terrified', 'panic', 'anxiety', 'worried', 'frightened', 'takot', 'natatakot', 'kinakabahan', 'kabado', 'nakakatakot'];
  const angerWords = ['angry', 'mad', 'furious', 'outraged', 'galit', 'nagagalit', 'poot', 'inis', 'yamot', 'bwisit'];
  const sadnessWords = ['sad', 'depressed', 'upset', 'devastated', 'heartbroken', 'grieving', 'malungkot', 'lungkot', 'kalungkutan', 'nakakalungkot'];
  const hopeWords = ['hope', 'hopeful', 'optimistic', 'faith', 'pag-asa', 'umaasa', 'naniniwalang'];
  const reliefWords = ['relief', 'relieved', 'thankful', 'grateful', 'ginhawa', 'natulungan', 'nakaligtas', 'ligtas', 'salamat'];
  const panicWords = ['panic', 'emergency', 'help', 'danger', 'urgent', 'tulong', 'saklolo', 'tabang', 'delikado'];
  
  if (fearWords.some(word => textLower.includes(word))) return 'Fear/Anxiety';
  if (angerWords.some(word => textLower.includes(word))) return 'Anger';
  if (sadnessWords.some(word => textLower.includes(word))) return 'Sadness';
  if (hopeWords.some(word => textLower.includes(word))) return 'Hope';
  if (reliefWords.some(word => textLower.includes(word))) return 'Relief';
  if (panicWords.some(word => textLower.includes(word))) return 'Panic';
  
  // Check for emojis
  if (text.includes('😭') || text.includes('😢') || text.includes('😓')) return 'Sadness';
  if (text.includes('😱') || text.includes('😨') || text.includes('😰')) return 'Fear/Anxiety';
  if (text.includes('😡') || text.includes('🤬') || text.includes('😠')) return 'Anger';
  if (text.includes('🙏') || text.includes('🤲') || text.includes('✨')) return 'Hope';
  if (text.includes('😌') || text.includes('😊') || text.includes('☺️')) return 'Relief';
  
  return 'Neutral';
}

// Helper function to find most common sentiment
function getMostCommonSentiment(posts) {
  const sentimentCounts = {};
  
  posts.forEach(post => {
    if (post.sentiment) {
      sentimentCounts[post.sentiment] = (sentimentCounts[post.sentiment] || 0) + 1;
    }
  });
  
  let mostCommonSentiment = null;
  let highestCount = 0;
  
  for (const sentiment in sentimentCounts) {
    if (sentimentCounts[sentiment] > highestCount) {
      highestCount = sentimentCounts[sentiment];
      mostCommonSentiment = sentiment;
    }
  }
  
  return mostCommonSentiment;
}

importData();