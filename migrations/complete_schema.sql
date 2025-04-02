-- Complete schema definition for Render deployment
-- This file contains all tables and columns required by the application
-- Last updated: April 2, 2025

-- Create tables if they don't exist
CREATE TABLE IF NOT EXISTS users (
  id SERIAL PRIMARY KEY,
  username TEXT NOT NULL UNIQUE,
  password TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  full_name TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  role TEXT DEFAULT 'user' NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id),
  token TEXT NOT NULL UNIQUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  expires_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS profile_images (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  role TEXT NOT NULL,
  image_url TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS sentiment_posts (
  id SERIAL PRIMARY KEY,
  text TEXT NOT NULL,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  source TEXT,
  language TEXT,
  sentiment TEXT NOT NULL,
  confidence REAL NOT NULL,
  location TEXT,
  disaster_type TEXT,
  file_id INTEGER,
  explanation TEXT,
  processed_by INTEGER REFERENCES users(id),
  ai_trust_message TEXT,
  runtime_order SERIAL
);

CREATE TABLE IF NOT EXISTS disaster_events (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  location TEXT,
  type TEXT NOT NULL,
  sentiment_impact TEXT,
  created_by INTEGER REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS analyzed_files (
  id SERIAL PRIMARY KEY,
  original_name TEXT NOT NULL,
  stored_name TEXT NOT NULL,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  record_count INTEGER NOT NULL,
  evaluation_metrics JSON,
  uploaded_by INTEGER REFERENCES users(id)
);

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

-- Add missing columns if tables already exist
ALTER TABLE IF EXISTS sentiment_posts 
  ADD COLUMN IF NOT EXISTS ai_trust_message TEXT,
  ADD COLUMN IF NOT EXISTS runtime_order SERIAL;

ALTER TABLE IF EXISTS sentiment_feedback 
  ADD COLUMN IF NOT EXISTS ai_trust_message TEXT,
  ADD COLUMN IF NOT EXISTS possible_trolling BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS training_error TEXT;

-- Fix column types if needed
ALTER TABLE IF EXISTS sentiment_posts ALTER COLUMN confidence TYPE REAL;