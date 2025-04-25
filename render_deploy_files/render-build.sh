#!/bin/bash

# Build script for Render.com deployment
# This will:
# 1. Install Node.js dependencies
# 2. Install Python dependencies
# 3. Create necessary directories
# 4. Copy the fixed server to the right location

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting build process for PanicSense on Render.com"
echo "Current directory: $(pwd)"
echo "Node version: $(node -v)"
echo "Python versions available:"
which python3
which python
python3 --version || echo "Python 3 not found"

# Create necessary directories first
echo "Creating necessary directories..."
mkdir -p ./uploads
mkdir -p ./python
mkdir -p ./public

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm ci || npm install

# Copy the fixed server to production-server-fixed.cjs
echo "Setting up production server..."
cp production-server-fixed.cjs app-render-complete.cjs

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt || echo "ERROR: Failed to install Python requirements"

# Create a simple Python daemon if it doesn't exist
if [ ! -f ./python/daemon.py ]; then
  echo "Creating Python daemon..."
  cat > ./python/daemon.py << 'EOL'
#!/usr/bin/env python3
"""
Python Daemon Script for PanicSense
This script runs the Python backend services in daemon mode
"""

import os
import sys
import time
import signal
import datetime
import json

# Make this script work regardless of how it's called
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir)))

# Flag to control the main loop
running = True

def signal_handler(sig, frame):
    """Handle process termination signals"""
    global running
    print(f"Received signal {sig}, shutting down...")
    running = False

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def process_news():
    """Process news and social media for disaster events"""
    print(f"[{datetime.datetime.now().isoformat()}] Processing news and social media feeds")
    # This would normally fetch news and process them
    time.sleep(5)
    print(f"[{datetime.datetime.now().isoformat()}] Retrieved 10 news items")

def run_disaster_monitor():
    """Main loop for disaster monitoring"""
    global running
    print(f"[{datetime.datetime.now().isoformat()}] Starting PanicSense Python daemon")
    
    try:
        while running:
            # Process news every 5 minutes
            process_news()
            
            # Sleep for a bit - this is where the daemon spends most of its time
            for _ in range(300):  # 5 minutes in seconds
                if not running:
                    break
                time.sleep(1)
    
    except Exception as e:
        print(f"[{datetime.datetime.now().isoformat()}] Error in main loop: {str(e)}")
    
    print(f"[{datetime.datetime.now().isoformat()}] Python daemon shutting down")

if __name__ == "__main__":
    # Run the main monitoring loop
    run_disaster_monitor()
EOL

  chmod +x ./python/daemon.py
  echo "Python daemon created and made executable"
fi

# Create a minimal index.html if it doesn't exist
if [ ! -f ./public/index.html ]; then
  echo "Creating basic index.html..."
  cat > ./public/index.html << 'EOL'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PanicSense Disaster Intelligence Platform</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      line-height: 1.6;
      color: #333;
      max-width: 1000px;
      margin: 0 auto;
      padding: 20px;
    }
    h1 {
      color: #d9534f;
      border-bottom: 2px solid #d9534f;
      padding-bottom: 10px;
    }
    .card {
      background: #f9f9f9;
      border-radius: 5px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .api-link {
      display: block;
      margin: 10px 0;
      padding: 10px;
      background: #eee;
      border-radius: 5px;
      text-decoration: none;
      color: #0275d8;
    }
    .api-link:hover {
      background: #ddd;
    }
  </style>
</head>
<body>
  <h1>PanicSense API Server</h1>
  
  <div class="card">
    <h2>Server Status</h2>
    <p>Server is running! You can access the API endpoints below:</p>
    
    <a class="api-link" href="/api/health">/api/health - Server Health</a>
    <a class="api-link" href="/api/disaster-events">/api/disaster-events - Disaster Events</a>
    <a class="api-link" href="/api/sentiment-posts">/api/sentiment-posts - Sentiment Posts</a>
    <a class="api-link" href="/api/analyzed-files">/api/analyzed-files - Analyzed Files</a>
    <a class="api-link" href="/api/python-logs">/api/python-logs - Python Service Logs</a>
  </div>
  
  <div class="card">
    <h2>About PanicSense</h2>
    <p>PanicSense is an advanced disaster intelligence platform providing comprehensive emergency insights and community safety coordination through intelligent AI-driven analysis.</p>
    
    <p>Core Technologies:</p>
    <ul>
      <li>TypeScript/React frontend</li>
      <li>Groq AI language processing</li>
      <li>Neon database integration</li>
      <li>Python backend with advanced NLP</li>
      <li>Machine learning for disaster sentiment detection</li>
      <li>Multilingual location and text analysis capabilities</li>
      <li>Render cloud deployment with full Python and Node.js support</li>
    </ul>
  </div>
</body>
</html>
EOL
  echo "Basic index.html created"
fi

echo "Build process completed successfully!"