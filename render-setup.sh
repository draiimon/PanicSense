#!/bin/bash

# This script will run when the application is deployed to Render
# It will handle database migrations and other setup tasks before starting the application

echo "=============================================="
echo "Starting setup for PanicSense PH on Render.com"
echo "$(date)"
echo "=============================================="

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p /data/uploads/temp

# Check database connection
echo "Checking database connection..."
max_retries=10
retry_count=0
while [ $retry_count -lt $max_retries ]; do
  # Remove DATABASE_URL= prefix if present
  DB_URL="${NEON_DATABASE_URL:-$DATABASE_URL}"
  if [[ "$DB_URL" == DATABASE_URL=* ]]; then
    DB_URL="${DB_URL#DATABASE_URL=}"
    echo "Fixed database URL format by removing prefix"
  fi
  
  if node -e "import pg from 'pg'; const { Pool } = pg; const pool = new Pool({ connectionString: '$DB_URL', ssl: { rejectUnauthorized: false } }); pool.query('SELECT NOW()', (err, res) => { if (err) { console.error(err); process.exit(1); } else { process.exit(0); } })" --input-type=module; then
    echo "Database connection successful!"
    break
  else
    retry_count=$((retry_count+1))
    echo "Database connection attempt $retry_count of $max_retries failed. Retrying in 5 seconds..."
    sleep 5
  fi
done

if [ $retry_count -eq $max_retries ]; then
  echo "ERROR: Could not connect to database after $max_retries attempts."
  exit 1
fi

# Run database migrations
echo "Running database migrations..."
npm run db:push

# Verify Python is available
echo "Verifying Python setup..."
if command -v python3 &> /dev/null; then
  echo "Python3 is available: $(python3 --version)"
else
  echo "WARNING: Python3 not found! The application may not function correctly."
fi

# Check if PyTorch is installed
echo "Checking PyTorch installation..."
if python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" &> /dev/null; then
  echo "PyTorch is installed correctly."
else
  echo "WARNING: PyTorch not found or not installed correctly. ML features may not work."
fi

echo "=============================================="
echo "All setup tasks complete! Application will now start."
echo "=============================================="