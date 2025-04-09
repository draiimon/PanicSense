#!/bin/bash

# This script is used to setup the application on Render

# Install Python dependencies
pip3 install -r python-requirements.txt

# Install NLTK data
mkdir -p /usr/share/nltk_data
python3 -m nltk.downloader -d /usr/share/nltk_data punkt vader_lexicon stopwords wordnet

# Create necessary directories
mkdir -p uploads/temp
mkdir -p attached_assets

# Run database migrations and initialization
node migrations/run-migrations.js

# Build the application
npm run build

echo "Render setup completed successfully!"