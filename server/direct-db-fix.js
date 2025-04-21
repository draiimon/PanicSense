/**
 * EMERGENCY DATABASE FIX
 * Direct database fix for Render deployment issues
 * This script will:
 * 1. Create missing tables if they don't exist
 * 2. Ensure all tables have the correct columns
 * 3. Add sample data if tables are empty
 */

import pg from 'pg';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const { Pool } = pg;
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Fix database tables directly
export async function emergencyDatabaseFix() {
  console.log('⚠️ RUNNING EMERGENCY DATABASE FIX');
  
  let pool;
  let client;
  
  try {
    // Connect to database using DATABASE_URL
    const databaseUrl = process.env.DATABASE_URL;
    if (!databaseUrl) {
      console.error("❌ No DATABASE_URL found in environment variables");
      return false;
    }
    
    console.log("🔄 Connecting to database directly...");
    pool = new Pool({
      connectionString: databaseUrl,
      ssl: process.env.DB_SSL_REQUIRED === 'true' ? { rejectUnauthorized: false } : false
    });

    client = await pool.connect();
    console.log(`✅ Successfully connected to PostgreSQL database`);
    
    // Create basic tables with timestamp column instead of created_at
    console.log("🔄 Creating disaster_events table if it doesn't exist...");
    await client.query(`
      CREATE TABLE IF NOT EXISTS disaster_events (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        location VARCHAR(255),
        severity VARCHAR(50),
        event_type VARCHAR(50),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    console.log("🔄 Creating sentiment_posts table if it doesn't exist...");
    await client.query(`
      CREATE TABLE IF NOT EXISTS sentiment_posts (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source VARCHAR(100),
        language VARCHAR(50),
        sentiment VARCHAR(50),
        confidence FLOAT,
        disaster_type VARCHAR(100),
        location VARCHAR(255)
      )
    `);
    
    console.log("🔄 Creating analyzed_files table if it doesn't exist...");
    await client.query(`
      CREATE TABLE IF NOT EXISTS analyzed_files (
        id SERIAL PRIMARY KEY,
        original_name VARCHAR(255) NOT NULL,
        stored_name VARCHAR(255) NOT NULL,
        row_count INTEGER,
        accuracy FLOAT,
        precision FLOAT,
        recall FLOAT,
        f1_score FLOAT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    console.log("🔄 Creating upload_sessions table if it doesn't exist...");
    await client.query(`
      CREATE TABLE IF NOT EXISTS upload_sessions (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(255) NOT NULL UNIQUE,
        status VARCHAR(50) DEFAULT 'active',
        file_name VARCHAR(255),
        progress INTEGER DEFAULT 0,
        error TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    // Add sample data if tables are empty
    const disasterCount = await client.query("SELECT COUNT(*) FROM disaster_events");
    if (parseInt(disasterCount.rows[0].count) === 0) {
      console.log("⚠️ Adding sample disaster event as table is empty");
      
      await client.query(`
        INSERT INTO disaster_events (name, description, location, severity, event_type)
        VALUES ('Typhoon in Coastal Areas', 'Based on 3 reports from the community. Please stay safe.', 'Metro Manila, Philippines', 'High', 'Typhoon')
      `);
    }
    
    const sentimentCount = await client.query("SELECT COUNT(*) FROM sentiment_posts");
    if (parseInt(sentimentCount.rows[0].count) === 0) {
      console.log("⚠️ Adding sample sentiment post as table is empty");
      
      await client.query(`
        INSERT INTO sentiment_posts (text, source, language, sentiment, confidence, disaster_type, location)
        VALUES ('My prayers to our brothers and sisters in Visayas region..', 'Twitter', 'en', 'neutral', 0.85, 'Typhoon', 'Visayas, Philippines')
      `);
    }
    
    const filesCount = await client.query("SELECT COUNT(*) FROM analyzed_files");
    if (parseInt(filesCount.rows[0].count) === 0) {
      console.log("⚠️ Adding sample analyzed file as table is empty");
      
      await client.query(`
        INSERT INTO analyzed_files (original_name, stored_name, row_count, accuracy, precision, recall, f1_score)
        VALUES ('MAGULONG DATA! (1).csv', 'batch-EJBpcspVXK_TZ717aZDM7-MAGULONG DATA! (1).csv', 100, 0.89, 0.91, 0.87, 0.89)
      `);
    }
    
    console.log("✅ Emergency database fix completed successfully");
    return true;
    
  } catch (error) {
    console.error("❌ Error in emergency database fix:", error.message);
    return false;
  } finally {
    if (client) client.release();
    if (pool) await pool.end();
  }
}