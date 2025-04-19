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

# Extract all API keys from file
import re
all_api_keys = re.findall(r'"(gsk_[^"]+)"', content)

print(f"Found {len(all_api_keys)} total API keys")

# Sample 10 keys from different parts of the list
def get_sample_keys(all_keys, sample_size=10):
    if len(all_keys) <= sample_size:
        return all_keys
    
    # Calculate steps to get evenly distributed samples
    step = len(all_keys) // sample_size
    
    # Get evenly distributed indices
    indices = [i * step for i in range(sample_size)]
    
    # Make sure we include the first and last keys
    if 0 not in indices:
        indices[0] = 0
    if len(all_keys) - 1 not in indices:
        indices[-1] = len(all_keys) - 1
        
    # Get the keys at these indices
    sampled_keys = [all_keys[i] for i in indices]
    
    return sampled_keys

# Test API keys
def test_key(key, index, max_retries=2):
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
    
    retries = 0
    backoff = 2
    
    while retries <= max_retries:
        try:
            print(f"Testing key #{index+1}/10: {short_key} (Attempt {retries+1})")
            
            response = requests.post(url, json=data, headers=headers, timeout=10)
            status = response.status_code
            
            if status == 200:
                print(f"✅ KEY #{index+1} WORKS: Status 200 OK")
                return True
                
            elif status == 429:  # Rate limit
                retry_after = int(response.headers.get('Retry-After', backoff))
                print(f"⏳ KEY #{index+1} RATE LIMITED: Waiting {retry_after}s before retry...")
                time.sleep(retry_after)
                retries += 1
                backoff *= 2
                continue
                
            else:
                try:
                    error_info = response.json()
                    error_message = error_info.get("error", {}).get("message", "Unknown error")
                    error_code = error_info.get("error", {}).get("code", "unknown")
                    print(f"❌ KEY #{index+1} FAILED: {error_message} (Code: {error_code})")
                except:
                    print(f"❌ KEY #{index+1} FAILED: Status {status} - Unable to parse response")
                return False
                
        except Exception as e:
            print(f"❌ KEY #{index+1} ERROR: {str(e)}")
            return False
    
    print(f"❌ KEY #{index+1} FAILED: Maximum retries exceeded")
    return False

def main():
    print("\n" + "="*60)
    print(" SAMPLING 10 GROQ API KEYS FOR QUICK TESTING")
    print("="*60)
    
    # Get 10 sample keys
    sample_keys = get_sample_keys(all_api_keys, 10)
    
    working_keys = 0
    broken_keys = 0
    
    for i, key in enumerate(sample_keys):
        if test_key(key, i):
            working_keys += 1
        else:
            broken_keys += 1
        
        # Sleep between tests to avoid global rate limiting
        if i < len(sample_keys) - 1:
            sleep_time = 3
            print(f"Sleeping for {sleep_time} seconds before next test...")
            time.sleep(sleep_time)
        
        print()  # Empty line for readability
    
    # Summary
    print("\n" + "="*60)
    print(" RESULTS SUMMARY")
    print("="*60)
    print(f"Total keys tested: 10")
    print(f"Working keys: {working_keys}")
    print(f"Broken keys: {broken_keys}")
    
    if working_keys == 0:
        print("\n❌ ALL TESTED KEYS FAILED.")
        print("There is strong evidence that all 45 keys are non-functional.")
        print("A new Groq API key is required.")
    else:
        print(f"\n✅ {working_keys} WORKING KEYS FOUND!")
        print(f"Some of the 45 keys may still be functional.")

if __name__ == "__main__":
    main()