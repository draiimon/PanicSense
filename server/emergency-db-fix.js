// EMERGENCY DATABASE FIX FOR RENDER DEPLOYMENT
// This script runs at application startup

const { Pool } = require('pg');

async function applyEmergencyFixes() {
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
    
    // Fix 1: Add ai_trust_message column to sentiment_posts
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

module.exports = { applyEmergencyFixes };