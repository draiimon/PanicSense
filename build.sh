#!/bin/bash

# Enhanced build script for Render deployment
echo "=== 🚀 PanicSense Render Build ==="

# Display environment information
echo "NODE_ENV: $NODE_ENV"
echo "Current directory: $(pwd)"
echo "Node version: $(node -v)"
echo "NPM version: $(npm -v)"

# Create upload folders
echo "=== 📁 Creating required folders ==="
mkdir -p uploads/{temp,data,profile_images}

# Install Vite and ESBuild globally to ensure they're available
echo "=== 📦 Installing build tools globally ==="
npm install -g vite esbuild

# Build client (copied from package.json script)
echo "=== 🏗️ Building app ==="
export NODE_ENV=production
npm run build

# Create requirements.txt if needed
if [ ! -f "requirements.txt" ]; then
  echo "=== 🐍 Creating Python requirements ==="
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
torch
tqdm
EOL
fi

# Install Python dependencies
echo "=== 🐍 Installing Python dependencies ==="
pip install -r render-requirements.txt || pip install -r requirements.txt

# Install NLTK data if needed
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

echo "=== ✅ Build complete ==="