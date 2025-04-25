#!/usr/bin/env python3
"""
Python Daemon Script for PanicSense
This script runs the Python backend services in daemon mode without requiring command-line arguments
"""

import os
import sys
import time
import signal
import json
import traceback
import random
from datetime import datetime

# Detect if we're running in a deployment environment
IS_PRODUCTION = os.environ.get('NODE_ENV', '').lower() == 'production'

# Print startup message
print(f"🐍 Python daemon starting at {datetime.now().isoformat()}")
print(f"🌍 Running in {'PRODUCTION' if IS_PRODUCTION else 'DEVELOPMENT'} mode")

# Set up signal handling for graceful shutdown
def signal_handler(sig, frame):
    """Handle process termination signals"""
    print(f"⚠️ Received termination signal {sig}, shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Main function to process news
def process_news():
    """Process news and social media for disaster events"""
    try:
        # Simulate processing news feeds
        print(f"📰 Processing news feeds at {datetime.now().isoformat()}")
        # In a real implementation, this would fetch and analyze news data
        time.sleep(2)  # Simulate work
        
        # Generate sample data
        disasters = ["Typhoon", "Earthquake", "Flood", "Fire", "Landslide"]
        locations = ["Manila", "Cebu", "Davao", "Baguio", "Iloilo"]
        
        # Simulate finding a disaster event
        if random.random() < 0.3:  # 30% chance of finding a disaster
            disaster = random.choice(disasters)
            location = random.choice(locations)
            print(f"🚨 Found potential {disaster} event in {location}")
            
            # In a real implementation, this would be saved to the database
            result = {
                "type": disaster,
                "location": location,
                "severity": random.choice(["Low", "Medium", "High", "Severe"]),
                "timestamp": datetime.now().isoformat()
            }
            print(f"✅ Processed event: {json.dumps(result)}")
        else:
            print("✓ No new disaster events detected")
            
    except Exception as e:
        print(f"❌ Error processing news: {str(e)}")
        traceback.print_exc()

# Main loop
def run_disaster_monitor():
    """Main loop for disaster monitoring"""
    print("🔄 Starting disaster monitoring loop")
    
    # Keep track of the last time we ran each task
    last_news_check = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Process news feeds every 15 minutes (or use the configured interval)
            news_interval = int(os.environ.get('NEWS_UPDATE_INTERVAL', 15)) * 60
            if current_time - last_news_check > news_interval:
                process_news()
                last_news_check = current_time
            
            # Sleep for 30 seconds between checks
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("⚠️ Received keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"❌ Error in disaster monitor loop: {str(e)}")
        traceback.print_exc()
        
        # In production, try to recover
        if IS_PRODUCTION:
            print("🔄 Attempting to restart monitoring loop in 60 seconds...")
            time.sleep(60)
            run_disaster_monitor()  # Restart the loop
        else:
            # In development, just exit with an error
            sys.exit(1)

# Start the monitor
if __name__ == "__main__":
    print("🚀 PanicSense Python daemon initialized")
    run_disaster_monitor()