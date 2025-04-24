#!/bin/bash

# Simple build script for Render deployment
echo "=== 🚀 PanicSense Render Build ==="

# Create upload folders
echo "=== 📁 Creating required folders ==="
mkdir -p uploads/{temp,data,profile_images}

# Build client (copied from package.json script)
echo "=== 🏗️ Building app ==="
npx vite build

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
pip install -r requirements.txt

# Install NLTK data if needed
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

echo "=== ✅ Build complete ==="