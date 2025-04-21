#!/bin/bash

# Start script for PanicSense Docker Containerized Application
# This script performs necessary initialization and starts the application

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting PanicSense application initialization...${NC}"

# Create required directories if they don't exist
mkdir -p /app/uploads/data
mkdir -p /app/uploads/profile_images
mkdir -p /app/uploads/temp

# Check if the environment variables are set
if [ -z "$DATABASE_URL" ]; then
  echo -e "${RED}ERROR: DATABASE_URL is not set. Please configure this in your Render environment variables.${NC}"
  exit 1
fi

# Wait for NeonDB to be available
echo -e "${YELLOW}Checking database connection...${NC}"
MAX_RETRIES=30
COUNT=0

while [ $COUNT -lt $MAX_RETRIES ]; do
  # Check if database is available using Node script
  node verify-db.js > /dev/null 2>&1
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}Database connection successful!${NC}"
    break
  fi
  
  COUNT=$((COUNT + 1))
  echo -e "${YELLOW}Waiting for database connection... ($COUNT/$MAX_RETRIES)${NC}"
  sleep 2
  
  if [ $COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}Failed to connect to database after $MAX_RETRIES attempts.${NC}"
    echo -e "${YELLOW}The application will start anyway, but may not function correctly.${NC}"
  fi
done

# Apply database migrations if needed
echo -e "${YELLOW}Performing database setup...${NC}"
node server/direct-db-fix.js

# Start the application
echo -e "${GREEN}Starting PanicSense application on port $PORT...${NC}"
exec node server.js