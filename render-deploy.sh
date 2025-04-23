#!/bin/bash
echo "==== Starting Comprehensive Render Deployment Build ===="

# Clean up any previous build artifacts
echo "Cleaning up previous builds..."
rm -rf dist client/dist

# Install all dependencies
echo "Installing dependencies..."
npm install

# Make sure vite is installed locally
echo "Ensuring vite is available..."
npm install vite

# Create directory structure needed by server
echo "Setting up file structure..."
mkdir -p client/dist
mkdir -p python

# Build the frontend
echo "Building frontend..."
npm run build || echo "Frontend build failed, continuing with backend only"

# Copy frontend files if build succeeded
if [ -d "dist" ]; then
  echo "Copying frontend files to client/dist..."
  cp -r dist/* client/dist/
fi

# Ensure Python files are in the right place
echo "Setting up Python files..."
if [ -d "server/python" ]; then
  echo "Copying Python files..."
  cp -r server/python/* python/
elif [ -d "python" ]; then
  echo "Python directory already exists, ensuring files are copied..."
  # This is a safety check
  if [ -f "server/python/process.py" ] && [ ! -f "python/process.py" ]; then
    cp -r server/python/* python/
  fi
fi

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
  echo "Creating requirements.txt..."
  cat > requirements.txt << 'EOL'
anthropic>=0.19.0
beautifulsoup4>=4.12.0
langdetect>=1.0.9
nltk>=3.8.0
numpy>=1.26.0
openai>=1.1.0
pandas>=2.0.0
python-dotenv>=1.0.0
pytz>=2024.1
requests>=2.31.0
scikit-learn>=1.3.0
torch>=2.0.0
tqdm>=4.65.0
transformers>=4.39.0
protobuf>=4.25.0
EOL
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt || echo "Python dependencies installation failed, continuing"

# Ensure Node.js server files are correctly installed
echo "Preparing server files..."
mkdir -p dist/server
cp -r server/* dist/server/
cp index.js dist/
cp hello-world.js dist/
cp package.json dist/

echo "==== Build Complete ===="