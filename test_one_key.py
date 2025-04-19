#!/usr/bin/env python3

import requests
import json
import sys
from api_key_tester import test_api_key

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_one_key.py <api_key>")
        return
    
    api_key = sys.argv[1]
    print("-" * 50)
    print(f"Testing API key: {api_key[:10]}***")
    
    working, message = test_api_key(api_key)
    
    if working:
        print(f"✅ SUCCESS: {message}")
    else:
        print(f"❌ FAILED: {message}")
    
    print("-" * 50)

if __name__ == "__main__":
    main()