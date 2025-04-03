#!/bin/bash

# Script to automatically run database migrations during deployment

echo "Starting PanicSense deployment quick fix script..."

# Create automatic migrations if needed
if [ -n "$DATABASE_URL" ]; then
  echo "Database URL found, running automatic migrations..."
  
  # Apply direct SQL fixes for missing columns
  echo "Running direct SQL migration for missing columns..."
  cat migrations/add_missing_columns.sql | psql "$DATABASE_URL"
  
  # Run database migrations using Drizzle ORM
  echo "Running npm run db:push to apply schema changes..."
  npm run db:push

  echo "Database migrations completed successfully!"
else
  echo "No DATABASE_URL found, skipping migrations."
fi

echo "Quick fix script completed."