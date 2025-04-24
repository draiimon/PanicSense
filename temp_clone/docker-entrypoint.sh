#!/bin/bash
set -e

# Activate Python virtual environment
source /app/venv/bin/activate

# Wait for database to be ready (if DATABASE_URL is provided)
if [ ! -z "$DATABASE_URL" ]; then
  echo "Waiting for PostgreSQL to be ready..."
  
  # Extract host and port from DATABASE_URL
  if [[ $DATABASE_URL == postgres* ]]; then
    # Extract host:port from the URL
    DB_HOST_PORT=$(echo $DATABASE_URL | sed -E 's/.*@([^\/]+).*/\1/')
    DB_HOST=$(echo $DB_HOST_PORT | cut -d':' -f1)
    DB_PORT=$(echo $DB_HOST_PORT | cut -d':' -f2)
    
    # Default to 5432 if no port specified
    if [ "$DB_HOST" = "$DB_PORT" ]; then
      DB_PORT=5432
    fi
    
    echo "Attempting to connect to database at $DB_HOST:$DB_PORT..."
    
    # Wait for database connection
    for i in {1..30}; do
      if pg_isready -h $DB_HOST -p $DB_PORT > /dev/null 2>&1; then
        echo "Database is ready!"
        break
      fi
      echo "Waiting for database connection... ($i/30)"
      sleep 2
      if [ $i -eq 30 ]; then
        echo "Warning: Could not connect to database, but continuing startup..."
      fi
    done
  else
    echo "DATABASE_URL doesn't appear to be a PostgreSQL URL, skipping connection check."
  fi
  
  # Run database setup
  echo "Setting up database tables..."
  node server/db-setup.js
  
  # Run emergency database fix if needed
  echo "Running database fixes if necessary..."
  NODE_PATH=. node -e "import('./server/direct-db-fix.js').then(m => m.emergencyDatabaseFix()).catch(e => console.error('Error running database fix:', e))"
fi

# Install NLTK data if needed
if [ ! -d "/app/venv/nltk_data" ]; then
  echo "Installing NLTK data..."
  python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
fi

# Execute the main command
echo "Starting application..."
exec "$@"