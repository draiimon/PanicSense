-- COMPLETE DATABASE SCHEMA FOR PANICSENSE

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

-- Create sessions table if it doesn't exist
CREATE TABLE IF NOT EXISTS sessions (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id),
  token TEXT NOT NULL UNIQUE,
  created_at TIMESTAMP DEFAULT NOW() NOT NULL,
  expires_at TIMESTAMP NOT NULL
);

-- Create sentiment_posts table if it doesn't exist
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

-- Create profile_images table if it doesn't exist
CREATE TABLE IF NOT EXISTS profile_images (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  role TEXT NOT NULL,
  image_url TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP DEFAULT NOW() NOT NULL
);

-- Create sentiment_feedback table if it doesn't exist
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
  user_id INTEGER REFERENCES users(id),
  ai_trust_message TEXT,
  possible_trolling BOOLEAN DEFAULT false
);

-- Create training_examples table if it doesn't exist
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

-- Create upload_sessions table if it doesn't exist
CREATE TABLE IF NOT EXISTS upload_sessions (
  id SERIAL PRIMARY KEY,
  session_id TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL DEFAULT 'active',
  file_id INTEGER REFERENCES analyzed_files(id) ON DELETE CASCADE,
  progress JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  server_start_timestamp TEXT
);

-- Now make sure all foreign keys are properly set up (if needed)
-- Note: Some foreign keys might fail if the referenced rows don't exist yet, but that's OK

-- Add references if they don't exist already
DO $$
BEGIN
  -- sentiment_posts.processed_by -> users.id 
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.table_constraints 
    WHERE constraint_name = 'sentiment_posts_processed_by_fkey'
  ) THEN
    BEGIN
      ALTER TABLE sentiment_posts 
      ADD CONSTRAINT sentiment_posts_processed_by_fkey 
      FOREIGN KEY (processed_by) REFERENCES users(id);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Skipping sentiment_posts_processed_by_fkey: %', SQLERRM;
    END;
  END IF;

  -- disaster_events.created_by -> users.id
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.table_constraints 
    WHERE constraint_name = 'disaster_events_created_by_fkey'
  ) THEN
    BEGIN
      ALTER TABLE disaster_events 
      ADD CONSTRAINT disaster_events_created_by_fkey 
      FOREIGN KEY (created_by) REFERENCES users(id);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Skipping disaster_events_created_by_fkey: %', SQLERRM;
    END;
  END IF;

  -- analyzed_files.uploaded_by -> users.id
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.table_constraints 
    WHERE constraint_name = 'analyzed_files_uploaded_by_fkey'
  ) THEN
    BEGIN
      ALTER TABLE analyzed_files 
      ADD CONSTRAINT analyzed_files_uploaded_by_fkey 
      FOREIGN KEY (uploaded_by) REFERENCES users(id);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Skipping analyzed_files_uploaded_by_fkey: %', SQLERRM;
    END;
  END IF;

  -- sentiment_posts.file_id -> analyzed_files.id
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.table_constraints 
    WHERE constraint_name = 'sentiment_posts_file_id_fkey'
  ) THEN
    BEGIN
      ALTER TABLE sentiment_posts 
      ADD CONSTRAINT sentiment_posts_file_id_fkey 
      FOREIGN KEY (file_id) REFERENCES analyzed_files(id);
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'Skipping sentiment_posts_file_id_fkey: %', SQLERRM;
    END;
  END IF;
END $$;