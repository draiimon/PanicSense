-- Add missing columns to sentiment_posts table
ALTER TABLE IF EXISTS sentiment_posts ADD COLUMN IF NOT EXISTS ai_trust_message text;

-- Add missing columns to sentiment_feedback table
ALTER TABLE IF EXISTS sentiment_feedback ADD COLUMN IF NOT EXISTS ai_trust_message text;
ALTER TABLE IF EXISTS sentiment_feedback ADD COLUMN IF NOT EXISTS possible_trolling boolean DEFAULT false;

-- Add missing tables if they don't exist
CREATE TABLE IF NOT EXISTS training_examples (
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

-- Ensure all tables are synced with the schema definition
CREATE TABLE IF NOT EXISTS sentiment_feedback (
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