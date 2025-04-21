#!/usr/bin/env python3
"""
Simple OpenAI API test script for Python
This script tests if the OpenAI API key is working correctly
"""

import os
import sys
from openai import OpenAI

# Initialize OpenAI client with API key from environment variable
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def test_openai():
    """Test if the OpenAI API key is working"""
    print("Testing OpenAI API connection...")
    
    try:
        # Simple API test - requesting a short message
        response = client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello from PanicSense! Can you confirm the OpenAI API is working?"}
            ],
            max_tokens=100
        )
        
        # Check if we got a valid response
        if response and response.choices and len(response.choices) > 0:
            print("✅ OpenAI API connection successful!")
            print(f"Response: {response.choices[0].message.content}")
            return True
        else:
            print("❌ OpenAI API returned an empty response")
            print(f"Response object: {response}")
            return False
    except Exception as e:
        print(f"❌ OpenAI API Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_openai()
    if success:
        print("✅ API test completed successfully")
        sys.exit(0)
    else:
        print("❌ API test failed")
        sys.exit(1)