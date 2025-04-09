// Simple database fix using ESM syntax for compatibility
// To be imported directly by index.ts

// Use ESM imports for compatibility
import pg from 'pg';
const { Pool } = pg;

// Simple emergency fix function
export async function simpleDbFix() {
  console.log('🚨 RUNNING SIMPLE DATABASE FIX 🚨');
  
  let pool;
  try {
    // Create a connection
    pool = new Pool({
      connectionString: process.env.DATABASE_URL,
      ssl: {
        rejectUnauthorized: false
      }
    });
    
    // Create user table first - needed for foreign keys
    await pool.query(`
    -- Create users table if it doesn't exist
    CREATE TABLE IF NOT EXISTS users (
      id SERIAL PRIMARY KEY,
      username TEXT NOT NULL UNIQUE,
      password TEXT NOT NULL,
      email TEXT NOT NULL UNIQUE,
      full_name TEXT NOT NULL,
      created_at TIMESTAMP DEFAULT NOW() NOT NULL,
      role TEXT DEFAULT 'user' NOT NULL
    );
    
    -- Create analyzed_files table if it doesn't exist
    CREATE TABLE IF NOT EXISTS analyzed_files (
      id SERIAL PRIMARY KEY,
      original_name TEXT NOT NULL,
      stored_name TEXT NOT NULL,
      timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
      record_count INTEGER NOT NULL,
      evaluation_metrics JSONB,
      uploaded_by INTEGER
    );

    -- Now we can create other tables with their foreign key references
    
    -- Create sessions table if it doesn't exist
    CREATE TABLE IF NOT EXISTS sessions (
      id SERIAL PRIMARY KEY,
      user_id INTEGER NOT NULL,
      token TEXT NOT NULL UNIQUE,
      created_at TIMESTAMP DEFAULT NOW() NOT NULL,
      expires_at TIMESTAMP NOT NULL
    );
    
    -- Create sentiment_posts table with all columns
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
    
    -- Create disaster_events table if it doesn't exist
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
    
    -- Create profile_images table if it doesn't exist
    CREATE TABLE IF NOT EXISTS profile_images (
      id SERIAL PRIMARY KEY,
      name TEXT NOT NULL,
      role TEXT NOT NULL,
      image_url TEXT NOT NULL,
      description TEXT,
      created_at TIMESTAMP DEFAULT NOW() NOT NULL
    );

    -- Create training_examples table
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
    
    -- Create sentiment_feedback table with all needed columns
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
    
    -- Create upload_sessions table for tracking file uploads
    CREATE TABLE IF NOT EXISTS upload_sessions (
      id SERIAL PRIMARY KEY,
      session_id TEXT NOT NULL UNIQUE,
      status TEXT NOT NULL DEFAULT 'active',
      file_id INTEGER,
      progress JSONB,
      created_at TIMESTAMP DEFAULT NOW(),
      updated_at TIMESTAMP DEFAULT NOW(),
      user_id INTEGER,
      server_start_timestamp TEXT
    );
    `);
    
    // Verify the tables were created
    const result = await pool.query(`
      SELECT 
        EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users') as users_exists,
        EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'analyzed_files') as analyzed_files_exists,
        EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sentiment_posts') as sentiment_posts_exists,
        EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'disaster_events') as disaster_events_exists,
        EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'training_examples') as training_examples_exists,
        EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sentiment_feedback') as feedback_exists,
        EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'upload_sessions') as upload_sessions_exists
    `);
    
    console.log('✅ SIMPLE FIX COMPLETE! Verification:', result.rows[0]);
    return true;
  } catch (err) {
    console.error('❌ SIMPLE DB FIX ERROR:', err.message);
    return false;
  } finally {
    if (pool) await pool.end();
  }
}