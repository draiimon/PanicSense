#!/bin/bash

# Setup GitHub token
GH_TOKEN="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"

echo "Step 1: Setting Git config..."
git config --global --add safe.directory "${PWD}"

echo "Step 2: Setup Git credentials..."
git config user.email "user@example.com" || echo "Failed to set user.email"
git config user.name "PanicSense Docker Setup" || echo "Failed to set user.name"

echo "Step 3: Adding files..."
git add Dockerfile docker-compose.yml start.sh .env.example docker-readme.md || echo "Failed to add files"

echo "Step 4: Committing changes..."
git commit -m "Add Docker configuration for local deployment" || echo "Failed to commit changes"

echo "Step 5: Setting remote URL with token..."
git remote set-url origin "https://${GH_TOKEN}@github.com/draiimon/PanicSense.git" || echo "Failed to set remote URL"

echo "Step 6: Pushing to GitHub..."
git push -u origin DockerPS || echo "Failed to push to GitHub"

echo "Done!"