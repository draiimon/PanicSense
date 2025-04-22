#!/bin/bash

# Start script for PanicSense Docker Containerized Application
# This script performs necessary initialization and starts the application

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting PanicSense application initialization...${NC}"

# ======== DEBUGGING INFORMATION ========
echo -e "${YELLOW}[DEBUG] System Information:${NC}"
uname -a
echo ""

echo -e "${YELLOW}[DEBUG] Python Information:${NC}"
python3 --version
which python3
echo ""

echo -e "${YELLOW}[DEBUG] NodeJS Information:${NC}"
node --version
which node
echo ""

echo -e "${YELLOW}[DEBUG] Key Environment Variables:${NC}"
env | grep -E "PYTHON|NODE|GROQ|ENABLE|DEBUG"
echo ""

echo -e "${YELLOW}[DEBUG] Current Working Directory:${NC}"
pwd
echo ""

echo -e "${YELLOW}[DEBUG] Application Directory Structure:${NC}"
ls -la /app
echo ""

echo -e "${YELLOW}[DEBUG] Python Directory Structure:${NC}"
ls -la /app/python
echo ""

# Create required directories if they don't exist
mkdir -p /app/uploads/data
mkdir -p /app/uploads/profile_images
mkdir -p /app/uploads/temp

echo -e "${YELLOW}[DEBUG] Uploads Directory Permissions:${NC}"
ls -la /app/uploads
echo ""

# Ensure proper permissions
chmod -R 777 /app/uploads
echo -e "${YELLOW}[DEBUG] Fixed Uploads Directory Permissions:${NC}"
ls -la /app/uploads
echo ""

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
echo -e "${YELLOW}Running emergency database fix directly...${NC}"
node server/direct-db-fix.js

# Apply Render-specific database fixes
echo -e "${YELLOW}Applying Render-specific database fixes...${NC}"
if [ "$RUNTIME_ENV" = "render" ]; then
  echo -e "${GREEN}Detected Render environment, applying specialized fixes${NC}"
  
  # Install PostgreSQL client tools if needed
  which psql > /dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing PostgreSQL client tools...${NC}"
    apt-get update && apt-get install -y postgresql-client
  fi
  
  # Run our specialized Render database fix script
  /app/render-db-fix.sh
  
  # Also run our Node.js fix as a backup
  echo -e "${YELLOW}Running secondary database fixes...${NC}"
  node -e "try { console.log('Fixing database tables...'); const { pool } = require('./server/db'); pool.query('ALTER TABLE analyzed_files ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP;'); pool.query('ALTER TABLE disaster_events ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP;'); } catch(e) { console.error('Error fixing tables:', e.message); }"
  
  echo -e "${GREEN}Render database fixes completed${NC}"
else
  echo -e "${YELLOW}Not running on Render, skipping specialized fixes${NC}"
fi

# Test Python operation
echo -e "${YELLOW}[DEBUG] Testing Python with NLTK import...${NC}"
python3 -c "import sys; print('Python path:', sys.path); import nltk; print('NLTK path:', nltk.data.path); import pandas; print('Pandas version:', pandas.__version__)" || echo -e "${RED}[ERROR] Failed to import Python libraries${NC}"

# Test Python script directly
echo -e "${YELLOW}[DEBUG] Testing process.py...${NC}"
cd /app/python && python3 -c "import process; print('Successfully imported process.py')" || echo -e "${RED}[ERROR] Failed to import process.py${NC}"
cd /app

# Check if news scraper is enabled
echo -e "${YELLOW}[DEBUG] News scraper status: ${ENABLE_SOCIAL_SCRAPER}${NC}"
if [ "$ENABLE_SOCIAL_SCRAPER" = "true" ]; then
  echo -e "${GREEN}[INFO] Social scraper is ENABLED${NC}"
else
  echo -e "${RED}[WARNING] Social scraper is DISABLED - News monitoring will not work!${NC}"
fi

# Check if Python service is enabled
echo -e "${YELLOW}[DEBUG] Python service status: ${PYTHON_SERVICE_ENABLED}${NC}"
if [ "$PYTHON_SERVICE_ENABLED" = "true" ]; then
  echo -e "${GREEN}[INFO] Python service is ENABLED${NC}"
  
  # Force NLTK data download (very important for text processing)
  echo -e "${YELLOW}Downloading NLTK data...${NC}"
  mkdir -p /root/nltk_data
  # Run with retry mechanism
  python3 -c "
import nltk
print('Starting NLTK downloads...')
try:
  nltk.download('punkt', quiet=True)
  print('✓ Downloaded punkt')
  nltk.download('stopwords', quiet=True)
  print('✓ Downloaded stopwords')
  nltk.download('wordnet', quiet=True)
  print('✓ Downloaded wordnet')
  print('All NLTK downloads completed successfully')
except Exception as e:
  print(f'Error downloading NLTK data: {e}')
" || echo -e "${RED}NLTK data download failed, but continuing...${NC}"

  # Verify NLTK setup
  echo -e "${YELLOW}Verifying NLTK setup...${NC}"
  python3 -c "
import sys
print('Python path:', sys.path)
try:
  import nltk
  print('✓ NLTK version:', nltk.__version__)
  print('✓ NLTK data path:', nltk.data.path)
except Exception as e:
  print(f'× NLTK error: {e}')
"
else
  echo -e "${RED}[WARNING] Python service is DISABLED - CSV processing and analysis will not work!${NC}"
fi

# Start the application
echo -e "${GREEN}Starting PanicSense application on port $PORT...${NC}"
exec node server.js
