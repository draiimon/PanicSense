#!/usr/bin/env python3

import requests
import json
import time
import sys
import random

# List of API keys to test (10 keys from different parts of the array)
api_keys = [
    # First 3 keys (start of list)
    "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",
    "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",
    "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",
    
    # Middle section (4 random keys)
    "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",
    "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",
    "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",
    "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",
    
    # Last 3 keys (end of list)
    "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",
    "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",
    "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"
]

def test_api_key(key, index):
    """Test if a Groq API key works and print results directly with rate limit handling."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    shortened_key = key[:10] + "***"
    
    # Try with rate limit handling (up to 2 retries)
    for attempt in range(1, 3):
        try:
            print(f"Testing key #{index+1}: {shortened_key} (Attempt {attempt})")
            response = requests.post(url, json=data, headers=headers, timeout=10)
            status = response.status_code
            
            if status == 200:
                print(f"✅ KEY #{index+1} WORKS: Status 200 OK\n")
                return True
            elif status == 429:  # Rate limit
                retry_after = int(response.headers.get('Retry-After', 5))
                print(f"⏳ KEY #{index+1} RATE LIMITED: Waiting {retry_after}s before retry...")
                time.sleep(retry_after)
                continue
            else:
                try:
                    error_info = response.json()
                    error_message = error_info.get("error", {}).get("message", "Unknown error")
                    error_code = error_info.get("error", {}).get("code", "unknown")
                    print(f"❌ KEY #{index+1} FAILED: {error_message} (Code: {error_code})\n")
                except:
                    print(f"❌ KEY #{index+1} FAILED: Status {status} - {response.text}\n")
                return False
        
        except Exception as e:
            print(f"❌ KEY #{index+1} ERROR: {str(e)}\n")
            if "timeout" in str(e).lower():
                print("Network timeout, will retry...")
                time.sleep(2)
                continue
            return False
    
    print(f"❌ KEY #{index+1} FAILED: Maximum retries exceeded\n")
    return False

def main():
    print("-" * 50)
    print("TESTING 10 REPRESENTATIVE GROQ API KEYS")
    print("-" * 50)
    
    working_count = 0
    failed_count = 0
    
    for i, key in enumerate(api_keys):
        success = test_api_key(key, i)
        if success:
            working_count += 1
        else:
            failed_count += 1
        
        # Add a small delay to avoid rate limiting (2 seconds)
        if i < len(api_keys) - 1:
            print(f"Sleeping for 3 seconds before next test...")
            time.sleep(3)
    
    print("-" * 50)
    print(f"SUMMARY: {working_count} working keys, {failed_count} broken keys")
    print("-" * 50)
    
    if working_count == 0:
        print("ALL TESTED KEYS ARE BROKEN.")
        print("A new API key is required.")
    else:
        print(f"{working_count} keys are working! We can continue using them.")

if __name__ == "__main__":
    main()