#!/bin/bash

# Complete Render.com build script for PanicSense
# Handles both Node.js and Python dependencies
set -e  # Exit on error

# Print environment information
echo "========== ENVIRONMENT INFO =========="
echo "Node version: $(node -v)"
echo "NPM version: $(npm -v)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# Install Python dependencies
echo "========== INSTALLING PYTHON DEPENDENCIES =========="
echo "Installing Python requirements..."
pip install anthropic beautifulsoup4 joblib langdetect matplotlib nltk numpy openai pandas python-dotenv pytz requests scikit-learn snscrape torch tqdm transformers

# Setup NLTK data
echo "Setting up NLTK data..."
python -m nltk.downloader punkt stopwords wordnet

# Install Node dependencies (including dev dependencies)
echo "========== INSTALLING NODE DEPENDENCIES =========="
echo "Installing all Node.js dependencies (including dev dependencies)..."
npm install --production=false

# Ensure vite is available
echo "========== CHECKING VITE =========="
if ! command -v ./node_modules/.bin/vite &> /dev/null; then
    echo "Vite not found in node_modules! Installing explicitly..."
    npm install --no-save vite
    echo "Vite explicitly installed: $(./node_modules/.bin/vite --version)"
fi

# Create necessary directories
echo "========== CREATING REQUIRED DIRECTORIES =========="
mkdir -p uploads
mkdir -p temp
mkdir -p logs

# Build the frontend and server
echo "========== BUILDING APPLICATION =========="
echo "Building the frontend and server..."
npm run build

# Verify build output
echo "========== VERIFYING BUILD =========="
if [ -d "dist/public" ]; then
    echo "✅ Frontend build successful! Files found in dist/public"
    ls -la dist/public
else
    echo "⚠️ Warning: Frontend build folder not found!"
fi

# Copy Python scripts to a specific location for the server
echo "========== PREPARING PYTHON SCRIPTS =========="
mkdir -p dist/python
cp -r python/* dist/python/
echo "Python scripts copied to dist/python"

echo "========== BUILD COMPLETE =========="
echo "Build completed successfully!"