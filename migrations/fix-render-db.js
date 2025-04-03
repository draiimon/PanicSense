// Direct database fix for Render deployment - bypasses Drizzle
// This script directly manipulates the database schema using SQL

import pg from 'pg';
const { Pool } = pg;
import dotenv from 'dotenv';
dotenv.config();

export async function fixRenderDatabase() {
  console.log('=== EMERGENCY DATABASE FIX SCRIPT ===');
  
  // Create a pool directly with the DATABASE_URL
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    // Use extended timeouts for slow connections
    statement_timeout: 60000,
    query_timeout: 60000,
    connectionTimeoutMillis: 30000,
  });

  try {
    console.log('Connected to database, applying emergency fixes...');
    
    // Execute each fix in separate try/catch blocks to ensure all attempts are made
    
    // Fix 1: Add ai_trust_message column to sentiment_posts
    try {
      console.log('Attempting to add ai_trust_message column...');
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
      console.log('ai_trust_message column fix completed');
    } catch (err) {
      console.error('Error adding ai_trust_message column:', err.message);
    }
    
    // Fix 2: Create training_examples table if it doesn't exist
    try {
      console.log('Attempting to create training_examples table...');
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
      console.log('training_examples table fix completed');
    } catch (err) {
      console.error('Error creating training_examples table:', err.message);
    }
    
    // Verify fixes
    try {
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
      
      console.log('Verification results:');
      console.log(`- ai_trust_message column exists: ${colCheck.rows[0].column_exists}`);
      console.log(`- training_examples table exists: ${tableCheck.rows[0].table_exists}`);
      
      if (colCheck.rows[0].column_exists && tableCheck.rows[0].table_exists) {
        console.log('✅ DATABASE FIXES SUCCESSFULLY APPLIED');
      } else {
        console.log('⚠️ SOME DATABASE FIXES FAILED');
      }
    } catch (err) {
      console.error('Error verifying fixes:', err.message);
    }
    
  } catch (err) {
    console.error('Fatal error:', err.message);
  } finally {
    await pool.end();
    console.log('Database connection closed');
  }
}

// Execute the fix immediately
fixRenderDatabase().catch(err => {
  console.error('Unhandled error in fix script:', err);
  process.exit(1);
});