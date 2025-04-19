#!/usr/bin/env python3

import requests
import json
import time
import sys
from api_key_tester import test_api_key

# Complete list of all 45 keys from process.py
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
    "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"
]

def main():
    print("Testing API keys...")
    print("-" * 50)
    
    working_keys = []
    broken_keys = []
    
    for i, key in enumerate(api_keys, 1):
        shortened_key = key[:10] + "***" 
        print(f"Testing key {i}/{len(api_keys)}: {shortened_key}")
        
        working, message = test_api_key(key)
        
        if working:
            working_keys.append((key, message))
            print(f"✅ {message}\n")
        else:
            broken_keys.append((key, message))
            print(f"❌ {message}\n")
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    print("-" * 50)
    print(f"Results: {len(working_keys)} working keys, {len(broken_keys)} broken keys")
    
    print("\nWorking keys:")
    for i, (key, message) in enumerate(working_keys, 1):
        shortened_key = key[:10] + "***"
        print(f"{i}. {shortened_key} - {message}")
    
    print("\nBroken keys:")
    for i, (key, message) in enumerate(broken_keys, 1):
        shortened_key = key[:10] + "***"
        error_type = message.split(" - ")[1] if " - " in message else message
        print(f"{i}. {shortened_key} - {error_type}")

if __name__ == "__main__":
    main()