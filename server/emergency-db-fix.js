// EMERGENCY DATABASE FIX FOR RENDER DEPLOYMENT
// This script runs at application startup

import pg from 'pg';
const { Pool } = pg;

export async function applyEmergencyFixes() {
  console.log('🚨 EMERGENCY DATABASE FIX - RUNNING AT STARTUP 🚨');
  
  let pool;
  try {
    // Create a connection pool
    pool = new Pool({
      connectionString: process.env.DATABASE_URL,
      // Extended timeouts for slow connections
      statement_timeout: 60000,
      query_timeout: 60000,
      connectionTimeoutMillis: 30000,
    });
    
    console.log('Connected to database, applying emergency fixes...');
    
    // First check if sentiment_posts table exists, and create it if not
    await pool.query(`
      CREATE TABLE IF NOT EXISTS sentiment_posts (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
        source TEXT,
        language TEXT,
        sentiment TEXT NOT NULL,
        confidence REAL NOT NULL,
        location TEXT,
        disaster_type TEXT,
        file_id INTEGER,
        explanation TEXT,
        processed_by INTEGER,
        ai_trust_message TEXT
      );
    `);
    console.log('✅ sentiment_posts table check completed');
    
    // Fix 1: Add ai_trust_message column to sentiment_posts if it doesn't exist
    await pool.query(`
      DO $$
      BEGIN
          IF NOT EXISTS (
              SELECT 1 
              FROM information_schema.columns 
              WHERE table_name = 'sentiment_posts' AND column_name = 'ai_trust_message'
          ) THEN
              ALTER TABLE sentiment_posts ADD COLUMN ai_trust_message text;
              RAISE NOTICE 'Added ai_trust_message column';
          ELSE
              RAISE NOTICE 'ai_trust_message column already exists';
          END IF;
      END $$;
    `);
    console.log('✅ ai_trust_message column fix applied');
    
    // Fix 2: Create training_examples table if it doesn't exist
    await pool.query(`
      CREATE TABLE IF NOT EXISTS training_examples (
          id SERIAL PRIMARY KEY,
          text TEXT NOT NULL,
          text_key TEXT NOT NULL,
          sentiment TEXT NOT NULL,
          language TEXT NOT NULL,
          confidence REAL DEFAULT 0.95 NOT NULL,
          created_at TIMESTAMP DEFAULT NOW(),
          updated_at TIMESTAMP DEFAULT NOW(),
          CONSTRAINT training_examples_text_unique UNIQUE(text),
          CONSTRAINT training_examples_text_key_unique UNIQUE(text_key)
      );
    `);
    console.log('✅ training_examples table fix applied');
    
    // Create other tables if they don't exist
    
    // disaster_events table
    await pool.query(`
      CREATE TABLE IF NOT EXISTS disaster_events (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
        location TEXT,
        type TEXT NOT NULL,
        sentiment_impact TEXT,
        created_by INTEGER
      );
    `);
    console.log('✅ disaster_events table check completed');
    
    // analyzed_files table
    await pool.query(`
      CREATE TABLE IF NOT EXISTS analyzed_files (
        id SERIAL PRIMARY KEY,
        original_name TEXT NOT NULL,
        stored_name TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
        record_count INTEGER NOT NULL,
        evaluation_metrics JSONB,
        uploaded_by INTEGER
      );
    `);
    console.log('✅ analyzed_files table check completed');
    
    // sentiment_feedback table
    await pool.query(`
      CREATE TABLE IF NOT EXISTS sentiment_feedback (
        id SERIAL PRIMARY KEY,
        original_post_id INTEGER,
        original_text TEXT NOT NULL,
        original_sentiment TEXT NOT NULL,
        corrected_sentiment TEXT DEFAULT '',
        corrected_location TEXT,
        corrected_disaster_type TEXT,
        trained_on BOOLEAN DEFAULT false,
        created_at TIMESTAMP DEFAULT NOW(),
        user_id INTEGER,
        ai_trust_message TEXT,
        possible_trolling BOOLEAN DEFAULT false
      );
    `);
    console.log('✅ sentiment_feedback table check completed');
    
    
    // Verify fixes
    const colCheck = await pool.query(`
      SELECT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'sentiment_posts' AND column_name = 'ai_trust_message'
      ) AS column_exists;
    `);
    
    const tableCheck = await pool.query(`
      SELECT EXISTS (
        SELECT 1 
        FROM information_schema.tables 
        WHERE table_name = 'training_examples'
      ) AS table_exists;
    `);
    
    const columnExists = colCheck.rows[0].column_exists;
    const tableExists = tableCheck.rows[0].table_exists;
    
    console.log(`Database verification: ai_trust_message column exists = ${columnExists}, training_examples table exists = ${tableExists}`);
    
    if (columnExists && tableExists) {
      console.log('✅ DATABASE SUCCESSFULLY FIXED AT STARTUP!');
      return true;
    } else {
      console.error('⚠️ DATABASE FIXES FAILED. THIS IS A CRITICAL ERROR.');
      return false;
    }
  } catch (err) {
    console.error('🚨 EMERGENCY DATABASE FIX FAILED:', err.message);
    return false;
  } finally {
    if (pool) {
      await pool.end();
    }
  }
}