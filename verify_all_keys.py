#!/usr/bin/env python3

import requests
import json
import time
import sys
import random
import os

# Get all API keys from process.py
with open('server/python/process.py', 'r') as f:
    content = f.read()

# Extract all API keys from file (any line containing gsk_)
import re
api_keys = re.findall(r'"(gsk_[^"]+)"', content)

print(f"Found {len(api_keys)} API keys to test")

# Function to test if key is working while handling rate limits
def test_key(key, index, max_retries=3, initial_backoff=2):
    """Test if a key works with retry logic for rate limiting"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": "Say hello"}]
    }
    
    # Show abbreviated key for privacy
    short_key = key[:10] + "***" 
    
    # Use retry mechanism with exponential backoff
    retries = 0
    backoff = initial_backoff
    
    while retries <= max_retries:
        try:
            print(f"Testing key #{index+1}/{len(api_keys)}: {short_key} (Attempt {retries+1})")
            
            response = requests.post(url, json=data, headers=headers, timeout=10)
            status = response.status_code
            
            # Check for different cases
            if status == 200:
                print(f"✅ KEY #{index+1} WORKS: Status 200 OK")
                return {"status": "working", "key": key}
                
            elif status == 429:  # Rate limit
                error_info = response.json()
                retry_after = int(response.headers.get('Retry-After', backoff))
                print(f"⏳ KEY #{index+1} RATE LIMITED: Waiting {retry_after}s before retry...")
                time.sleep(retry_after)
                retries += 1
                backoff *= 2  # Exponential backoff
                continue
                
            else:
                try:
                    error_info = response.json()
                    error_message = error_info.get("error", {}).get("message", "Unknown error")
                    error_code = error_info.get("error", {}).get("code", "unknown")
                    print(f"❌ KEY #{index+1} FAILED: {error_message} (Code: {error_code})")
                    return {"status": "broken", "key": key, "error": f"{error_message} (Code: {error_code})"}
                except:
                    print(f"❌ KEY #{index+1} FAILED: Status {status} - Unable to parse response")
                    return {"status": "broken", "key": key, "error": f"HTTP {status}"}
                
        except requests.exceptions.Timeout:
            print(f"⏱️ KEY #{index+1} TIMEOUT: Request timed out, retrying...")
            retries += 1
            time.sleep(backoff)
            backoff *= 2
            
        except Exception as e:
            print(f"❌ KEY #{index+1} ERROR: {str(e)}")
            return {"status": "broken", "key": key, "error": str(e)}
    
    # If we exceeded retries
    print(f"❌ KEY #{index+1} FAILED: Maximum retries exceeded (likely rate limited)")
    return {"status": "possibly_rate_limited", "key": key, "error": "Maximum retries exceeded"}

def main():
    print("\n" + "="*60)
    print(" COMPREHENSIVE GROQ API KEY VERIFICATION")
    print("="*60)
    
    results = {
        "working": [],
        "broken": [],
        "possibly_rate_limited": []
    }
    
    for i, key in enumerate(api_keys):
        result = test_key(key, i)
        results[result["status"]].append(result)
        
        # Sleep between tests to avoid global rate limiting
        sleep_time = 5
        if i < len(api_keys) - 1:  # Don't sleep after the last key
            print(f"Sleeping for {sleep_time} seconds before next test...")
            time.sleep(sleep_time)
        
        print()  # Empty line for readability
    
    # Summary
    print("\n" + "="*60)
    print(" RESULTS SUMMARY")
    print("="*60)
    print(f"Total keys tested: {len(api_keys)}")
    print(f"Working keys: {len(results['working'])}")
    print(f"Broken keys: {len(results['broken'])}")
    print(f"Possibly rate limited: {len(results['possibly_rate_limited'])}")
    
    # Show working keys (if any)
    if results['working']:
        print("\nWorking keys:")
        for r in results['working']:
            print(f"- {r['key'][:10]}***")
    
    # Create updated keys file
    updated_keys = list(api_keys)  # Start with a copy of original keys
    
    # Remove broken keys
    if results['broken']:
        broken_keys_set = set(r['key'] for r in results['broken'])
        updated_keys = [k for k in updated_keys if k not in broken_keys_set]
    
    # Show broken keys removed
    print("\nErrors encountered:")
    error_counts = {}
    for r in results['broken']:
        error = r['error']
        if error not in error_counts:
            error_counts[error] = 0
        error_counts[error] += 1
    
    for error, count in error_counts.items():
        print(f"- {error}: {count} keys")
    
    # Show how many keys we kept 
    print(f"\nRecommendation: Keep {len(updated_keys)} keys, remove {len(api_keys) - len(updated_keys)} broken keys")
    
    # Create new file with only working keys
    if len(updated_keys) != len(api_keys):
        # Only create the file if there's a change
        with open('working_groq_keys.txt', 'w') as f:
            for key in updated_keys:
                f.write(f'"{key}",\n')
        print("\n✅ Created working_groq_keys.txt with all functioning keys")
    
    if not results['working'] and not results['possibly_rate_limited']:
        print("\n❌ WARNING: No working keys found. A new Groq API key is required.")

if __name__ == "__main__":
    main()