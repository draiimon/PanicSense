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
pnpm run db:push

# Build the application
pnpm run build

echo "Render setup completed successfully!"