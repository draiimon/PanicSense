#!/bin/bash
set -e

# Configure git if needed
git config --global user.email "user@example.com"
git config --global user.name "PanicSense Docker Setup"

# Setup GitHub token
GH_TOKEN="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"

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Create and checkout the DockerPS branch
git checkout -b DockerPS
echo "Created and switched to DockerPS branch"

# Add the Docker files
git add Dockerfile docker-compose.yml start.sh .env.example docker-readme.md
echo "Added Docker configuration files"

# Commit the changes
git commit -m "Add Docker configuration for local deployment"
echo "Committed changes"

# Update remote URL with token
git remote set-url origin https://${GH_TOKEN}@github.com/draiimon/PanicSense.git
echo "Updated remote URL with token"

# Push to GitHub with the provided token
git push -u origin DockerPS
echo "Pushed changes to GitHub in the DockerPS branch"

# Switch back to original branch
git checkout ${CURRENT_BRANCH}
echo "Switched back to $CURRENT_BRANCH branch"

echo "Successfully pushed Docker configuration to GitHub in the DockerPS branch!"