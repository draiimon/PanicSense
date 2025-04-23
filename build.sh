#!/bin/bash

# Simple build script for Render FREE TIER deployment
echo "=== 🚀 PanicSense FREE TIER Build Process ==="

# Install dependencies 
npm install

# Build the app
npm run build

# Create required folders
mkdir -p dist/public uploads/{temp,data,profile_images} python

# Copy frontend and Python files
if [ -d "client/dist" ]; then
  cp -r client/dist/* dist/public/ 2>/dev/null || true
elif [ -d "public" ]; then
  cp -r public/* dist/public/ 2>/dev/null || true
fi

# Copy Python files if needed
if [ -d "server/python" ]; then
  cp -r server/python/* python/ 2>/dev/null || true
fi

# Create Python requirements if missing
if [ ! -f "requirements.txt" ]; then
  cat > requirements.txt << EOL
anthropic
beautifulsoup4
langdetect
nltk
numpy
openai
pandas
python-dotenv
pytz
requests
scikit-learn
snscrape
tqdm
EOL
fi

# Installing minimal Python dependencies
pip install -r requirements.txt

echo "=== ✅ Build complete - FREE TIER READY ==="