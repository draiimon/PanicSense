#!/usr/bin/env python
"""
Python Daemon Script for PanicSense
This script runs the Python backend services in daemon mode without requiring command-line arguments
"""

import os
import sys
import time
import json
import logging
import signal
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Import the process module
try:
    from process import DisasterSentimentBackend
    logging.info("Successfully imported disaster backend")
except Exception as e:
    logging.error(f"Failed to import disaster backend: {e}")
    sys.exit(1)

# Create the backend
backend = DisasterSentimentBackend()
logging.info("Initialized disaster sentiment backend")

# Flag to control the main loop
running = True

def signal_handler(sig, frame):
    """Handle process termination signals"""
    global running
    logging.info(f"Received signal {sig}, shutting down...")
    running = False

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def process_news():
    """Process news and social media for disaster events"""
    try:
        logging.info("Starting news processing cycle")
        # This would normally query news sources and process them
        logging.info("News processing completed")
        return True
    except Exception as e:
        logging.error(f"Error processing news: {e}")
        return False

def run_disaster_monitor():
    """Main loop for disaster monitoring"""
    logging.info("Starting disaster monitoring service")
    
    cycle_count = 0
    
    while running:
        try:
            cycle_count += 1
            logging.info(f"Monitoring cycle {cycle_count} started")
            
            # Process news every 15 minutes (adjust as needed)
            if cycle_count % 15 == 0:
                process_news()
            
            # Check for any pending file analysis requests
            # This would normally check the database for files to analyze
            
            # Sleep for 1 minute between cycles
            logging.info(f"Monitoring cycle {cycle_count} completed, sleeping for 60 seconds")
            for _ in range(60):
                if not running:
                    break
                time.sleep(1)
                
        except Exception as e:
            logging.error(f"Error in monitoring cycle: {e}")
            # Sleep for 10 seconds before retrying after an error
            time.sleep(10)
    
    logging.info("Disaster monitoring service stopped")

if __name__ == "__main__":
    logging.info("PanicSense Python daemon starting...")
    
    # Test the backend
    test_text = "There's a massive flood in Manila! Water is rising quickly. #emergency"
    try:
        result = backend.analyze_sentiment(test_text)
        logging.info(f"Backend test: Successfully analyzed test text")
        logging.info(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']}")
        logging.info(f"Disaster type: {result['disasterType']}, Location: {result['location']}")
    except Exception as e:
        logging.error(f"Backend test failed: {e}")
    
    # Start the main monitoring loop
    run_disaster_monitor()
    
    logging.info("PanicSense Python daemon exiting normally")