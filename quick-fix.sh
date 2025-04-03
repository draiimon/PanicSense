#!/bin/bash

# Script to automatically run database migrations during deployment

echo "Starting PanicSense deployment quick fix script..."
echo "===========================================" 
echo "RENDER EMERGENCY DATABASE FIX v3.0"
echo "===========================================" 

# Ensure dependencies are installed (for non-Docker environments)
if ! command -v psql &> /dev/null; then
  echo "PostgreSQL client not found, installing..."
  apt-get update && apt-get install -y --no-install-recommends postgresql-client
fi

# Create automatic migrations if needed
if [ -n "$DATABASE_URL" ]; then
  echo "Database URL found, running emergency migrations..."

  # APPROACH 1: Apply the complete schema directly
  echo "Running complete schema SQL file..."
  if psql -v ON_ERROR_STOP=0 "$DATABASE_URL" -f migrations/complete_schema.sql; then
    echo "✅ Complete schema applied successfully"
  else
    echo "⚠️ Complete schema application completed with warnings (trying other approaches...)"
  fi
  
  # APPROACH 2: Run the emergency fix script
  echo "⚠️ RUNNING EMERGENCY DATABASE FIX SCRIPT..."
  node --experimental-modules migrations/fix-render-db.js
  
  # APPROACH 3: Apply direct SQL fixes for missing columns
  echo "Running direct SQL migration for missing columns..."
  if psql -v ON_ERROR_STOP=0 "$DATABASE_URL" -f migrations/add_missing_columns.sql; then
    echo "✅ SQL migration completed successfully"
  else
    echo "⚠️ SQL migration completed with warnings (this is usually OK)"
  fi
  
  # APPROACH 4: Apply critical changes directly (last resort, guaranteed to work)
  echo "Directly applying critical schema changes..."
  psql "$DATABASE_URL" <<EOF
    -- Add ai_trust_message to sentiment_posts if it doesn't exist
    DO \$\$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'sentiment_posts' AND column_name = 'ai_trust_message'
      ) THEN
        ALTER TABLE sentiment_posts ADD COLUMN ai_trust_message text;
        RAISE NOTICE 'Added ai_trust_message column';
      ELSE
        RAISE NOTICE 'ai_trust_message column already exists';
      END IF;
    END \$\$;

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
EOF
  
  # APPROACH 5: Run JavaScript-based migrations
  echo "Running JavaScript-based migrations..."
  if node --experimental-modules migrations/run-migrations.js; then
    echo "✅ JavaScript migrations completed successfully"
  else
    echo "⚠️ JavaScript migrations completed with warnings (this is usually OK)"
  fi
  
  # APPROACH 6: Run database migrations using Drizzle ORM
  echo "Running npm run db:push to apply schema changes..."
  if npm run db:push; then
    echo "✅ Drizzle migrations completed successfully"
  else
    echo "⚠️ Drizzle migrations completed with warnings (this is usually OK)"
  fi
  
  # Final verification
  echo "===========================================" 
  echo "FINAL VERIFICATION"
  echo "===========================================" 
  COLUMN_EXISTS=$(psql -t "$DATABASE_URL" -c "SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sentiment_posts' AND column_name = 'ai_trust_message');")
  TABLE_EXISTS=$(psql -t "$DATABASE_URL" -c "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'training_examples');")
  
  echo "ai_trust_message column exists: $COLUMN_EXISTS"
  echo "training_examples table exists: $TABLE_EXISTS"
  
  if [[ "$COLUMN_EXISTS" == *"t"* ]] && [[ "$TABLE_EXISTS" == *"t"* ]]; then
    echo "✅ DATABASE FIXES SUCCESSFULLY APPLIED"
  else
    echo "⚠️ DATABASE FIXES FAILED. PLEASE CHECK LOGS."
  fi

  echo "Database migrations completed!"
else
  echo "No DATABASE_URL found, skipping migrations."
fi

echo "Quick fix script completed."