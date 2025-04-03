// Direct database migration using JavaScript and Drizzle ORM
import { drizzle } from 'drizzle-orm/node-postgres';
import pkg from 'pg';
const { Pool } = pkg;
import { sql } from 'drizzle-orm';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Define the database connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

// Create a Drizzle client
const db = drizzle(pool);

async function runMigrations() {
  console.log('Starting direct JavaScript migrations...');

  try {
    // Manually check and add ai_trust_message column to sentiment_posts
    console.log('Adding ai_trust_message column to sentiment_posts if it doesn\'t exist...');
    await db.execute(sql`
      DO $$
      BEGIN
          IF NOT EXISTS (
              SELECT 1 FROM information_schema.columns 
              WHERE table_name = 'sentiment_posts' AND column_name = 'ai_trust_message'
          ) THEN
              ALTER TABLE sentiment_posts ADD COLUMN ai_trust_message text;
              RAISE NOTICE 'Added ai_trust_message column to sentiment_posts table';
          ELSE
              RAISE NOTICE 'ai_trust_message column already exists in sentiment_posts table';
          END IF;
      END $$;
    `);

    // Check if sentiment_feedback table exists and create if needed
    console.log('Checking sentiment_feedback table...');
    await db.execute(sql`
      DO $$
      BEGIN
          IF EXISTS (
              SELECT 1 FROM information_schema.tables 
              WHERE table_name = 'sentiment_feedback'
          ) THEN
              -- Table exists, check for missing columns
              IF NOT EXISTS (
                  SELECT 1 FROM information_schema.columns 
                  WHERE table_name = 'sentiment_feedback' AND column_name = 'ai_trust_message'
              ) THEN
                  ALTER TABLE sentiment_feedback ADD COLUMN ai_trust_message text;
                  RAISE NOTICE 'Added ai_trust_message column to sentiment_feedback table';
              END IF;
              
              IF NOT EXISTS (
                  SELECT 1 FROM information_schema.columns 
                  WHERE table_name = 'sentiment_feedback' AND column_name = 'possible_trolling'
              ) THEN
                  ALTER TABLE sentiment_feedback ADD COLUMN possible_trolling boolean DEFAULT false;
                  RAISE NOTICE 'Added possible_trolling column to sentiment_feedback table';
              END IF;
          ELSE
              -- Create table
              CREATE TABLE sentiment_feedback (
                  id serial PRIMARY KEY NOT NULL,
                  original_post_id integer REFERENCES sentiment_posts(id) ON DELETE CASCADE,
                  original_text text NOT NULL,
                  original_sentiment text NOT NULL,
                  corrected_sentiment text DEFAULT '',
                  corrected_location text,
                  corrected_disaster_type text,
                  trained_on boolean DEFAULT false,
                  created_at timestamp DEFAULT now(),
                  user_id integer REFERENCES users(id),
                  ai_trust_message text,
                  possible_trolling boolean DEFAULT false
              );
              RAISE NOTICE 'Created sentiment_feedback table';
          END IF;
      END $$;
    `);

    // Create training_examples table if it doesn't exist
    console.log('Checking training_examples table...');
    await db.execute(sql`
      DO $$
      BEGIN
          IF NOT EXISTS (
              SELECT 1 FROM information_schema.tables 
              WHERE table_name = 'training_examples'
          ) THEN
              CREATE TABLE training_examples (
                  id serial PRIMARY KEY NOT NULL,
                  text text NOT NULL,
                  text_key text NOT NULL,
                  sentiment text NOT NULL,
                  language text NOT NULL,
                  confidence real NOT NULL DEFAULT 0.95,
                  created_at timestamp DEFAULT now(),
                  updated_at timestamp DEFAULT now(),
                  CONSTRAINT training_examples_text_unique UNIQUE(text),
                  CONSTRAINT training_examples_text_key_unique UNIQUE(text_key)
              );
              RAISE NOTICE 'Created training_examples table';
          ELSE
              RAISE NOTICE 'training_examples table already exists';
          END IF;
      END $$;
    `);

    // Verify database structure
    const aiTrustMessageExists = await db.execute(sql`
      SELECT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'sentiment_posts' AND column_name = 'ai_trust_message'
      );
    `);
    
    const trainingExamplesExists = await db.execute(sql`
      SELECT EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'training_examples'
      );
    `);

    console.log('Database verification:');
    console.log('- ai_trust_message column in sentiment_posts exists:', aiTrustMessageExists.rows[0].exists);
    console.log('- training_examples table exists:', trainingExamplesExists.rows[0].exists);

    console.log('Direct JavaScript migrations completed successfully!');
  } catch (error) {
    console.error('Error running migrations:', error);
    throw error;
  } finally {
    // Close the database connection
    await pool.end();
  }
}

// Run the migrations
runMigrations().catch(err => {
  console.error('Migration failed:', err);
  process.exit(1);
});