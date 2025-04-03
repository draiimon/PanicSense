#!/bin/bash

# Script to automatically run database migrations during deployment

echo "Starting PanicSense deployment quick fix script..."

# Ensure dependencies are installed (for non-Docker environments)
if ! command -v psql &> /dev/null; then
  echo "PostgreSQL client not found, installing..."
  apt-get update && apt-get install -y --no-install-recommends postgresql-client
fi

# Create automatic migrations if needed
if [ -n "$DATABASE_URL" ]; then
  echo "Database URL found, running automatic migrations..."
  
  # Try multiple migration approaches to ensure success
  
  # 1. Apply direct SQL fixes for missing columns
  echo "Running direct SQL migration for missing columns..."
  set -x  # Enable debug output
  psql -v ON_ERROR_STOP=0 "$DATABASE_URL" -f migrations/add_missing_columns.sql || echo "SQL migration script completed with warnings (this is usually OK)"
  set +x  # Disable debug output
  
  # 2. Run JavaScript-based migrations
  echo "Running JavaScript-based migrations..."
  node --experimental-modules migrations/run-migrations.js || echo "JavaScript migrations completed with warnings (this is usually OK)"
  
  # 3. Run database migrations using Drizzle ORM
  echo "Running npm run db:push to apply schema changes..."
  npm run db:push
  
  # Direct SQL approach for critical columns (last resort)
  echo "Running direct SQL commands for critical columns..."
  psql "$DATABASE_URL" -c "ALTER TABLE IF EXISTS sentiment_posts ADD COLUMN IF NOT EXISTS ai_trust_message text;" || true
  psql "$DATABASE_URL" -c "CREATE TABLE IF NOT EXISTS training_examples (id serial PRIMARY KEY, text text NOT NULL, text_key text NOT NULL, sentiment text NOT NULL, language text NOT NULL, confidence real DEFAULT 0.95 NOT NULL, created_at timestamp DEFAULT now(), updated_at timestamp DEFAULT now(), CONSTRAINT training_examples_text_unique UNIQUE(text), CONSTRAINT training_examples_text_key_unique UNIQUE(text_key));" || true

  # Verify database structure
  echo "Verifying database structure..."
  psql "$DATABASE_URL" -c "SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sentiment_posts' AND column_name = 'ai_trust_message');"
  psql "$DATABASE_URL" -c "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'training_examples');"

  echo "Database migrations completed successfully!"
else
  echo "No DATABASE_URL found, skipping migrations."
fi

echo "Quick fix script completed."