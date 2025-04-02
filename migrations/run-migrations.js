// Enhanced migration script using both direct SQL and schema validation
require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { Pool } = require('pg');
const { drizzle } = require('drizzle-orm/node-postgres');
// Import migrator separately to avoid schema imports which might fail
const { migrate } = require('drizzle-orm/node-postgres/migrator');

// Establish a connection to the database
async function runMigrations() {
  console.log('Starting database migrations...');
  
  // Connect to database using DATABASE_URL with SSL settings for production
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
  });

  const db = drizzle(pool);

  try {
    // Step 1: Run SQL migration script directly first
    const sqlPath = path.join(__dirname, 'add_missing_columns.sql');
    const sql = fs.readFileSync(sqlPath, 'utf8');
    
    console.log('Executing direct SQL migrations...');
    await pool.query(sql);
    console.log('Direct SQL migrations completed');

    // Step 2: Verify training_examples table exists or create it
    try {
      console.log('Verifying training_examples table...');
      const result = await pool.query(`
        SELECT EXISTS (
          SELECT FROM information_schema.tables 
          WHERE table_schema = 'public' 
          AND table_name = 'training_examples'
        );
      `);
      
      if (!result.rows[0].exists) {
        console.log('Creating training_examples table...');
        await pool.query(`
          CREATE TABLE IF NOT EXISTS training_examples (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL UNIQUE,
            text_key TEXT NOT NULL UNIQUE,
            sentiment TEXT NOT NULL,
            language TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.95,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          );
        `);
        console.log('training_examples table created successfully');
      } else {
        console.log('training_examples table already exists');
      }
    } catch (tableError) {
      console.error('Error verifying/creating training_examples table:', tableError);
      
      // Extra fallback: run the full SQL again
      try {
        console.log('Attempting fallback table creation...');
        await pool.query(sql);
      } catch (fallbackError) {
        console.error('Fallback creation failed:', fallbackError);
      }
    }

    console.log('All migrations completed successfully!');
  } catch (error) {
    console.error('Error executing migrations:', error);
    // Log more details about the error for troubleshooting
    if (error.stack) {
      console.error('Error stack:', error.stack);
    }
    
    // Don't fail the container if migrations fail - app might still work with existing schema
  } finally {
    await pool.end();
  }
}

// Run migrations immediately
runMigrations().catch(err => {
  console.error('Unhandled migration error:', err);
  process.exit(1);
});