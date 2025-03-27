#!/bin/bash

echo "Starting database migration..."

# Run drizzle-kit push with non-interactive mode and force options
echo "Pushing database schema..."
npx drizzle-kit push:pg --force

# Check if the push was successful
if [ $? -eq 0 ]; then
  echo "Schema successfully pushed to the database"
else
  echo "Failed to push schema to the database"
  exit 1
fi

# Restart the application
echo "Restarting application..."
npm run dev &

echo "Migration completed successfully!"