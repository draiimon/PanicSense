// Script to update existing sentiment posts with disaster types
import { db } from './server/db.js';
import { sql } from 'drizzle-orm';

async function updateDisasterTypes() {
  console.log('Updating disaster types for existing posts...');
  
  // Set all nulls to "Uncategorized Disaster"
  await db.execute(sql`
    UPDATE sentiment_posts 
    SET "disasterType" = 'Uncategorized Disaster' 
    WHERE "disasterType" IS NULL
  `);
  
  // Update based on keywords in the text
  await db.execute(sql`
    UPDATE sentiment_posts 
    SET "disasterType" = 'Earthquake' 
    WHERE "disasterType" = 'Uncategorized Disaster' 
    AND (
      text ILIKE '%earthquake%' OR 
      text ILIKE '%tremor%' OR 
      text ILIKE '%seismic%' OR 
      text ILIKE '%lindol%'
    )
  `);
  
  await db.execute(sql`
    UPDATE sentiment_posts 
    SET "disasterType" = 'Typhoon' 
    WHERE "disasterType" = 'Uncategorized Disaster' 
    AND (
      text ILIKE '%typhoon%' OR 
      text ILIKE '%storm%' OR 
      text ILIKE '%cyclone%' OR 
      text ILIKE '%bagyo%'
    )
  `);
  
  await db.execute(sql`
    UPDATE sentiment_posts 
    SET "disasterType" = 'Flood' 
    WHERE "disasterType" = 'Uncategorized Disaster' 
    AND (
      text ILIKE '%flood%' OR 
      text ILIKE '%baha%' OR 
      text ILIKE '%submerged%'
    )
  `);
  
  await db.execute(sql`
    UPDATE sentiment_posts 
    SET "disasterType" = 'Fire' 
    WHERE "disasterType" = 'Uncategorized Disaster' 
    AND (
      text ILIKE '%fire%' OR 
      text ILIKE '%sunog%' OR 
      text ILIKE '%burning%' OR 
      text ILIKE '%flame%'
    )
  `);
  
  await db.execute(sql`
    UPDATE sentiment_posts 
    SET "disasterType" = 'Landslide' 
    WHERE "disasterType" = 'Uncategorized Disaster' 
    AND (
      text ILIKE '%landslide%' OR 
      text ILIKE '%mudslide%' OR 
      text ILIKE '%pagguho%'
    )
  `);
  
  console.log('Successfully updated disaster types!');
  process.exit(0);
}

updateDisasterTypes().catch(err => {
  console.error('Error updating disaster types:', err);
  process.exit(1);
});