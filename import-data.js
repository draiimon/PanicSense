import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';
import { db } from './server/db.js';
import { sentimentPosts, disasterEvents } from './shared/schema.js';

async function importData() {
  try {
    console.log('Starting data import...');
    
    // Import test-data-variant.csv
    const variantDataPath = path.join(process.cwd(), 'test-data-variant.csv');
    console.log(`Reading from ${variantDataPath}`);
    
    if (fs.existsSync(variantDataPath)) {
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
        
        await db.insert(sentimentPosts).values(sentimentPost);
      }
      
      console.log(`Imported ${records.length} records from test-data-variant.csv`);
    } else {
      console.log('File test-data-variant.csv does not exist');
    }
    
    // Import test-data-with-location.csv
    const locationDataPath = path.join(process.cwd(), 'test-data-with-location.csv');
    console.log(`Reading from ${locationDataPath}`);
    
    if (fs.existsSync(locationDataPath)) {
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
        
        await db.insert(sentimentPosts).values(sentimentPost);
      }
      
      console.log(`Imported ${records.length} records from test-data-with-location.csv`);
    } else {
      console.log('File test-data-with-location.csv does not exist');
    }
    
    // Create some disaster events based on the imported data
    const uniqueDisasters = new Set();
    const posts = await db.select().from(sentimentPosts);
    
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
        
        await db.insert(disasterEvents).values(event);
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