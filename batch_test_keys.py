#!/usr/bin/env python3

import requests
import json
import time
import sys

# Fourth set of 15 of 45 keys from process.py (keys 31-45)
api_keys = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
]

def test_api_key(key, index):
    """Test if a Groq API key works and print results directly."""
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
    
    try:
        print(f"Testing key #{index+1}: {shortened_key}")
        response = requests.post(url, json=data, headers=headers)
        status = response.status_code
        
        if status == 200:
            print(f"✅ KEY #{index+1} WORKS: Status 200 OK\n")
            return True
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
        return False

def main():
    print("-" * 50)
    print("TESTING FIRST 10 GROQ API KEYS")
    print("-" * 50)
    
    working_count = 0
    failed_count = 0
    
    for i, key in enumerate(api_keys):
        success = test_api_key(key, i)
        if success:
            working_count += 1
        else:
            failed_count += 1
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
    
    print("-" * 50)
    print(f"SUMMARY: {working_count} working keys, {failed_count} broken keys")
    print("-" * 50)

if __name__ == "__main__":
    main()