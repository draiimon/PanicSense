#!/usr/bin/env python3

import os
import sys
import json
import time
import requests
import csv
from datetime import datetime

# Create CSV file with header
def create_csv_file():
    with open("all_api_keys_status.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Key Number", "Status", "Response Time (s)", "Working"])
    print("Created CSV file with header")

# Check a single API key and append result to CSV
def check_key(key_number, api_key):
    print(f"Testing API key #{key_number}...", end="", flush=True)
    
    # Sample text for testing
    text = "May lindol sa Davao. Tumayo ako agad at tumakbo palabas ng bahay."
    
    # API call setup
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_message = """You are a disaster sentiment analysis expert. Categorize text into: 'Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', or 'Neutral'. Respond in JSON format: {"sentiment": "category"}"""
    
    data = {
        "model": "gemma2-9b-it",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ],
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    # Status to record
    status = "UNKNOWN"
    response_time = 0
    working = "NO"
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response_time = round(time.time() - start_time, 2)
        
        if response.status_code == 200:
            status = "SUCCESS"
            working = "YES"
            print(f" ✅ SUCCESS (Response time: {response_time}s)")
        elif response.status_code == 429:
            status = "RATE LIMITED"
            working = "NO"
            print(f" ❌ RATE LIMITED (Response time: {response_time}s)")
        else:
            status = f"ERROR {response.status_code}"
            working = "NO"
            print(f" ❌ ERROR {response.status_code} (Response time: {response_time}s)")
    
    except Exception as e:
        status = f"EXCEPTION: {str(e)[:30]}..."
        working = "NO"
        print(f" ❌ EXCEPTION: {str(e)[:30]}...")
    
    # Save to CSV
    with open("all_api_keys_status.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([key_number, status, response_time, working])

def main():
    # Create the CSV file with header
    create_csv_file()
    
    # Get all the API keys
    api_keys = [
        "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",  # Key 1
        "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",  # Key 2
        "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",  # Key 3
        "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",  # Key 4
        "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",  # Key 5
        "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",  # Key 6
        "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",  # Key 7
        "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",  # Key 8
        "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",  # Key 9
        "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",  # Key 10
        "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",  # Key 11
        "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",  # Key 12
        "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",  # Key 13
        "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",  # Key 14
        "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",  # Key 15
        "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",  # Key 16
        "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",  # Key 17
        "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",  # Key 18
        "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",  # Key 19
        "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",  # Key 20
        "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",  # Key 21
        "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",  # Key 22
        "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",  # Key 23
        "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",  # Key 24
        "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",  # Key 25
        "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",  # Key 26
        "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",  # Key 27     
        "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",  # Key 28
        "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",  # Key 29
        "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",  # Key 30
        "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",  # Key 31
        "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",  # Key 32
        "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",  # Key 33
        "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",  # Key 34
        "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",  # Key 35
        "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",  # Key 36
        "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",  # Key 37
        "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",  # Key 38
        "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",  # Key 39
        "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",  # Key 40
        "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",  # Key 41
        "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",  # Key 42
        "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",  # Key 43
        "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",  # Key 44
        "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"   # Key 45
    ]
    
    # Check each key
    for i, api_key in enumerate(api_keys, 1):
        check_key(i, api_key)
        # Sleep to avoid rate limiting between checks
        if i < len(api_keys):
            print("Waiting 2 seconds before next check...")
            time.sleep(2)
    
    # Print completion message and summary
    print("\nCheck complete! Results saved to all_api_keys_status.csv")
    print(f"Checked {len(api_keys)} API keys.")

if __name__ == "__main__":
    main()