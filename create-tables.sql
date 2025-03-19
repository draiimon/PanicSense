
-- Create users table
CREATE TABLE IF NOT EXISTS users (
  id SERIAL PRIMARY KEY,
  username TEXT NOT NULL UNIQUE,
  password TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  full_name TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  role TEXT NOT NULL DEFAULT 'user'
);

-- Create sessions table
CREATE TABLE IF NOT EXISTS sessions (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id),
  token TEXT NOT NULL UNIQUE,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMP NOT NULL
);

-- Create sentiment_posts table
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
  processed_by INTEGER REFERENCES users(id)
);

-- Create disaster_events table
CREATE TABLE IF NOT EXISTS disaster_events (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
  location TEXT,
  type TEXT NOT NULL,
  sentiment_impact TEXT,
  created_by INTEGER REFERENCES users(id)
);

-- Create analyzed_files table
CREATE TABLE IF NOT EXISTS analyzed_files (
  id SERIAL PRIMARY KEY,
  original_name TEXT NOT NULL,
  stored_name TEXT NOT NULL,
  timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
  record_count INTEGER NOT NULL,
  evaluation_metrics JSON,
  uploaded_by INTEGER REFERENCES users(id)
);

-- Create profile_images table
CREATE TABLE IF NOT EXISTS profile_images (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  role TEXT NOT NULL,
  image_url TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
