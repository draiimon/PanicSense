#!/bin/bash

# Script to push changes to main branch

echo "Setting up git credentials"
git config --global credential.helper store
echo "https://x-access-token:$GITHUB_TOKEN@github.com" > ~/.git-credentials

echo "Configuring git username and email"
git config --global user.name "Draiimon"
git config --global user.email "draiimon@users.noreply.github.com"

echo "Setting git pull strategy to rebase"
git config pull.rebase true

echo "Attempting to force push changes to main branch"
git push -f origin main

echo "Push completed."