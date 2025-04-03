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
    });
    
    // Just create tables with all columns included
    await pool.query(`
    -- sentiment_posts table with all columns
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

    -- training_examples table
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
    
    -- sentiment_feedback table with all needed columns
    CREATE TABLE IF NOT EXISTS sentiment_feedback (
      id SERIAL PRIMARY KEY,
      original_post_id INTEGER REFERENCES sentiment_posts(id) ON DELETE CASCADE,
      original_text TEXT NOT NULL,
      original_sentiment TEXT NOT NULL,
      corrected_sentiment TEXT DEFAULT '',
      corrected_location TEXT,
      corrected_disaster_type TEXT,
      trained_on BOOLEAN DEFAULT false,
      created_at TIMESTAMP DEFAULT NOW(),
      user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
      ai_trust_message TEXT,
      possible_trolling BOOLEAN DEFAULT false
    );
    `);
    
    // Verify
    const result = await pool.query(`
      SELECT 
        EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sentiment_posts' AND column_name = 'ai_trust_message') as col_exists,
        EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'training_examples') as table_exists,
        EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sentiment_feedback') as feedback_exists
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