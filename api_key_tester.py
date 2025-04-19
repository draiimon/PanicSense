#!/usr/bin/env python3

import requests
import json
import time
import sys

def test_api_key(key):
    """Test if a Groq API key works."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        status = response.status_code
        
        if status == 200:
            return True, "API key works properly"
        else:
            try:
                error_info = response.json()
                error_message = error_info.get("error", {}).get("message", "Unknown error")
                error_code = error_info.get("error", {}).get("code", "unknown")
                return False, f"Error: {status} - {error_message} (Code: {error_code})"
            except:
                return False, f"Error: {status} - {response.text}"
    
    except Exception as e:
        return False, f"Exception: {str(e)}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python api_key_tester.py <api_key>")
        return
    
    api_key = sys.argv[1]
    working, message = test_api_key(api_key)
    
    if working:
        print(f"✅ API KEY WORKS: {message}")
    else:
        print(f"❌ API KEY FAILED: {message}")

if __name__ == "__main__":
    main()