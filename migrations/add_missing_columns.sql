-- Add missing columns in Render deployment
ALTER TABLE IF EXISTS sentiment_posts ADD COLUMN IF NOT EXISTS ai_trust_message TEXT;
ALTER TABLE IF EXISTS sentiment_feedback ADD COLUMN IF NOT EXISTS ai_trust_message TEXT;
ALTER TABLE IF EXISTS sentiment_feedback ADD COLUMN IF NOT EXISTS possible_trolling BOOLEAN DEFAULT FALSE;
ALTER TABLE IF EXISTS sentiment_feedback ADD COLUMN IF NOT EXISTS training_error TEXT;

-- Create training_examples table if not exists
CREATE TABLE IF NOT EXISTS training_examples (
  id SERIAL PRIMARY KEY,
  text TEXT NOT NULL UNIQUE,
  text_key TEXT NOT NULL UNIQUE,
  sentiment TEXT NOT NULL,
  language TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.95,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);