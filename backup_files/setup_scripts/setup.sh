#!/bin/bash

# Setup script for Disaster Sentiment Analysis Platform
# This script helps with GitHub setup and containerization

echo "========== Disaster Sentiment Analysis Platform Setup =========="
echo "This script will help you set up the project for GitHub and Docker"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo -e "\n==== Checking for required tools ===="

if ! command_exists git; then
  echo "❌ Git is not installed. Please install Git first."
  exit 1
else
  echo "✅ Git is installed"
fi

if ! command_exists docker; then
  echo "❌ Docker is not installed. Containerization will not be available."
  DOCKER_AVAILABLE=false
else
  echo "✅ Docker is installed"
  DOCKER_AVAILABLE=true
fi

if ! command_exists docker-compose; then
  if [ "$DOCKER_AVAILABLE" = true ]; then
    echo "❌ Docker Compose is not installed. Containerization will not be fully available."
    COMPOSE_AVAILABLE=false
  fi
else
  echo "✅ Docker Compose is installed"
  COMPOSE_AVAILABLE=true
fi

# GitHub setup
echo -e "\n==== GitHub Setup ===="
echo "Do you want to initialize a new Git repository? (y/n)"
read init_git

if [ "$init_git" = "y" ] || [ "$init_git" = "Y" ]; then
  if [ -d .git ]; then
    echo "Git repository already exists"
  else
    git init
    echo "Git repository initialized"
  fi
  
  echo "Do you want to add a remote GitHub repository? (y/n)"
  read add_remote
  
  if [ "$add_remote" = "y" ] || [ "$add_remote" = "Y" ]; then
    echo "Enter your GitHub repository URL (e.g., https://github.com/username/repo.git):"
    read repo_url
    
    if [ -n "$repo_url" ]; then
      git remote add origin "$repo_url"
      echo "Remote repository added as 'origin'"
    else
      echo "No URL provided, skipping remote setup"
    fi
  fi
  
  echo "Do you want to make an initial commit? (y/n)"
  read make_commit
  
  if [ "$make_commit" = "y" ] || [ "$make_commit" = "Y" ]; then
    git add .
    git commit -m "Initial commit: Disaster Sentiment Analysis Platform"
    echo "Initial commit created"
  fi
else
  echo "Skipping Git initialization"
fi

# Docker setup
if [ "$DOCKER_AVAILABLE" = true ] && [ "$COMPOSE_AVAILABLE" = true ]; then
  echo -e "\n==== Docker Setup ===="
  echo "Do you want to build and run the Docker containers? (y/n)"
  read run_docker
  
  if [ "$run_docker" = "y" ] || [ "$run_docker" = "Y" ]; then
    echo "Building and starting Docker containers..."
    docker-compose up --build -d
    echo "Docker containers are now running in the background"
    echo "- Application: http://localhost:5000"
    echo "- pgAdmin (database management): http://localhost:8080"
    echo "  - Email: admin@example.com"
    echo "  - Password: admin"
  else
    echo "Skipping Docker setup"
    echo "To build and run the containers later, use: docker-compose up --build"
  fi
fi

echo -e "\n========== Setup Complete =========="
echo "You can now start working with your Disaster Sentiment Analysis Platform!"
echo "- To push to GitHub: git push origin main"
if [ "$DOCKER_AVAILABLE" = true ] && [ "$COMPOSE_AVAILABLE" = true ]; then
  echo "- To stop Docker containers: docker-compose down"
  echo "- To restart Docker containers: docker-compose up -d"
fi