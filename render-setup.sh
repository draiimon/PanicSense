#!/bin/bash

# This script is used to setup the application on Render

# Install Python dependencies
pip3 install -r python-requirements.txt

# Install NLTK data
mkdir -p /usr/share/nltk_data
python3 -m nltk.downloader -d /usr/share/nltk_data punkt vader_lexicon stopwords wordnet

# Ensure uploads directory exists
mkdir -p uploads
touch uploads/.gitkeep

# Run database migrations
npm run db:push

# Build the application
npm run build

# Fix ES module issue in package.json (if needed)
if grep -q '"type": "module"' package.json; then
  echo "Ensuring package is properly set for ES modules..."
  echo -e '#!/usr/bin/env node\n\nimport("./index.js");' > dist/wrapper.js
  chmod +x dist/wrapper.js
  echo "Created ES module wrapper"
fi

echo "Render setup completed successfully!"