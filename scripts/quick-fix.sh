#!/bin/bash
# Quick fix script for Render deployments
# Run this in the Render shell tab for existing deployments
# This script will create the training_examples table and add the ai_trust_message column

echo "Connecting to database..."
echo "Running schema initialization..."

psql $DATABASE_URL << 'EOF'
-- Create training_examples table if not exists
CREATE TABLE IF NOT EXISTS training_examples (
  id SERIAL PRIMARY KEY,
  text TEXT NOT NULL UNIQUE,
  text_key TEXT NOT NULL UNIQUE,
  sentiment TEXT NOT NULL,
  language TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.95,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add ai_trust_message column to sentiment_posts if it doesn't exist
ALTER TABLE IF EXISTS sentiment_posts ADD COLUMN IF NOT EXISTS ai_trust_message TEXT;

-- Add runtime_order column to sentiment_posts if it doesn't exist
ALTER TABLE IF EXISTS sentiment_posts ADD COLUMN IF NOT EXISTS runtime_order SERIAL;

-- Create sentiment_feedback table if not exists
CREATE TABLE IF NOT EXISTS sentiment_feedback (
  id SERIAL PRIMARY KEY,
  original_post_id INTEGER REFERENCES sentiment_posts(id) ON DELETE CASCADE,
  original_text TEXT NOT NULL,
  original_sentiment TEXT NOT NULL,
  corrected_sentiment TEXT DEFAULT '',
  corrected_location TEXT,
  corrected_disaster_type TEXT,
  trained_on BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  user_id INTEGER REFERENCES users(id),
  ai_trust_message TEXT,
  possible_trolling BOOLEAN DEFAULT FALSE,
  training_error TEXT
);
EOF

echo "Database fix complete! You may need to restart your application."