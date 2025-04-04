#!/usr/bin/env python3

import os
import sys
import json
import time
import requests
from datetime import datetime
import csv

# Test prompt to validate API key functionality
TEST_TEXT = "May lindol sa Davao. Tumayo ako agad at tumakbo palabas ng bahay."

def check_api_key(api_key, key_number):
    """
    Test a single API key functionality and return its status
    """
    print(f"Testing API key #{key_number}... ", end="", flush=True)
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_message = """You are a disaster sentiment analysis expert for the Philippines.
    Your task is to DEEPLY ANALYZE THE FULL CONTEXT of each message and categorize it into one of: 
    'Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', or 'Neutral'.
    Choose ONLY ONE category and provide a confidence score (0.0-1.0) and brief explanation.
    
    Respond ONLY in JSON format: {"sentiment": "category", "confidence": score, "explanation": "explanation", "disasterType": "type", "location": "location"}"""
    
    data = {
        "model": "gemma2-9b-it",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": TEST_TEXT}
        ],
        "temperature": 0.1,
        "max_tokens": 500,
        "top_p": 1,
        "stream": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data, timeout=15)
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        
        if response.status_code == 200:
            # Extract the sentiment from the response
            resp_data = response.json()
            if "choices" in resp_data and resp_data["choices"]:
                content = resp_data["choices"][0]["message"]["content"]
                
                # Try to parse JSON response - with multiple methods
                try:
                    # Method 1: Direct parsing
                    try:
                        result = json.loads(content)
                    except:
                        # Method 2: Extract JSON object using regex
                        import re
                        json_match = re.search(r'{.*}', content, re.DOTALL)
                        if json_match:
                            result = json.loads(json_match.group(0))
                        else:
                            # Method 3: Clean up the content and try again
                            # Replace common issues like single quotes with double quotes
                            fixed_content = content.replace("'", '"')
                            # Try to find and extract a JSON-like structure
                            json_match = re.search(r'{.*}', fixed_content, re.DOTALL)
                            if json_match:
                                result = json.loads(json_match.group(0))
                            else:
                                raise ValueError("No JSON structure found")
                    
                    sentiment = result.get("sentiment", "Unknown")
                    confidence = result.get("confidence", 0.0)
                    print(f"✅ SUCCESS - Response time: {response_time}s - Sentiment: {sentiment} ({confidence})")
                    return {
                        "key_number": key_number,
                        "status": "SUCCESS",
                        "response_time": response_time,
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "error": None
                    }
                except Exception as e:
                    # API call succeeded but JSON parsing failed with all methods
                    print(f"✅ SUCCESS (API RESPONSE GOOD) - Response time: {response_time}s - JSON parsing issue: {str(e)}")
                    return {
                        "key_number": key_number,
                        "status": "SUCCESS",
                        "response_time": response_time,
                        "sentiment": "Unknown",
                        "confidence": 0.0,
                        "error": f"JSON parsing issue: {str(e)}"
                    }
            else:
                print(f"⚠️ PARTIAL SUCCESS - Response time: {response_time}s - No choices in response")
                return {
                    "key_number": key_number,
                    "status": "PARTIAL SUCCESS",
                    "response_time": response_time,
                    "sentiment": "Unknown",
                    "confidence": 0.0,
                    "error": "No choices in response"
                }
        elif response.status_code == 429:
            print(f"❌ RATE LIMITED - Response time: {response_time}s")
            return {
                "key_number": key_number,
                "status": "RATE LIMITED",
                "response_time": response_time,
                "sentiment": "Unknown",
                "confidence": 0.0,
                "error": "Rate limited (429)"
            }
        else:
            print(f"❌ FAILED - Status code: {response.status_code} - Response time: {response_time}s")
            return {
                "key_number": key_number,
                "status": "FAILED",
                "response_time": response_time,
                "sentiment": "Unknown",
                "confidence": 0.0,
                "error": f"HTTP error {response.status_code}: {response.text}"
            }
    except Exception as e:
        print(f"❌ ERROR - {str(e)}")
        return {
            "key_number": key_number,
            "status": "ERROR",
            "response_time": 0,
            "sentiment": "Unknown",
            "confidence": 0.0,
            "error": str(e)
        }

def save_results_to_csv(results, filename="api_key_check_results.csv"):
    """
    Save the results to a CSV file
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['key_number', 'status', 'response_time', 'sentiment', 'confidence', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nResults saved to {filename}")

def main():
    # Get all API keys from environment variables
    api_keys = []
    
    # First check environment variables API_KEY_1 through API_KEY_45
    for i in range(1, 46):
        key_name = f"API_KEY_{i}"
        api_key = os.getenv(key_name)
        if api_key:
            api_keys.append((api_key, i))
    
    # Backup method - check hardcoded keys
    if not api_keys:
        hardcoded_keys = [
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
        
        for i, key in enumerate(hardcoded_keys, 1):
            api_keys.append((key, i))
    
    print(f"Found {len(api_keys)} API keys to test")
    
    # Check all API keys with a delay between each check to avoid rate limiting
    results = []
    
    for api_key, key_number in api_keys:
        # Check the key
        result = check_api_key(api_key, key_number)
        results.append(result)
        
        # Wait a moment between requests to avoid rate limiting
        time.sleep(2)
    
    # Print summary
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    rate_limited_count = sum(1 for r in results if r["status"] == "RATE LIMITED")
    failed_count = sum(1 for r in results if r["status"] in ["FAILED", "ERROR"])
    partial_count = sum(1 for r in results if r["status"] == "PARTIAL SUCCESS")
    
    print("\n=== SUMMARY ===")
    print(f"Total keys tested: {len(results)}")
    print(f"Successful: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"Rate limited: {rate_limited_count} ({rate_limited_count/len(results)*100:.1f}%)")
    print(f"Failed: {failed_count} ({failed_count/len(results)*100:.1f}%)")
    print(f"Partial success: {partial_count} ({partial_count/len(results)*100:.1f}%)")
    
    # Save results to CSV
    save_results_to_csv(results)
    
    print("\nRate-limited keys:")
    rate_limited_keys = [r["key_number"] for r in results if r["status"] == "RATE LIMITED"]
    if rate_limited_keys:
        print(f"Keys: {', '.join(map(str, rate_limited_keys))}")
    else:
        print("None!")
    
    # Return code based on results
    if rate_limited_count + failed_count > 0:
        return 1
    else:
        return 0

if __name__ == "__main__":
    # Allow for testing specific key ranges
    import argparse
    parser = argparse.ArgumentParser(description='Test API keys')
    parser.add_argument('--start', type=int, default=1, help='Starting key number (1-45)')
    parser.add_argument('--end', type=int, default=10, help='Ending key number (1-45)')
    
    args = parser.parse_args()
    
    # Check arguments
    if args.start < 1:
        args.start = 1
    if args.end > 45:
        args.end = 45
    if args.start > args.end:
        print(f"Error: Start ({args.start}) must be less than or equal to end ({args.end})")
        sys.exit(1)
    
    # Modify the main function to only test keys in the specified range
    def ranged_main():
        # Get all API keys from environment variables
        api_keys = []
        
        # First check environment variables API_KEY_1 through API_KEY_45
        for i in range(args.start, args.end + 1):
            key_name = f"API_KEY_{i}"
            api_key = os.getenv(key_name)
            if api_key:
                api_keys.append((api_key, i))
        
        # Backup method - check hardcoded keys
        if not api_keys:
            hardcoded_keys = [
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
            
            # Filter for only keys in the specified range
            filtered_keys = hardcoded_keys[args.start - 1:args.end]
            for i, key in enumerate(filtered_keys, args.start):
                api_keys.append((key, i))
        
        print(f"Testing API keys #{args.start}-{args.end}")
        print(f"Found {len(api_keys)} API keys to test")
        
        # Check all API keys with a delay between each check to avoid rate limiting
        results = []
        
        for api_key, key_number in api_keys:
            # Check the key
            result = check_api_key(api_key, key_number)
            results.append(result)
            
            # Wait a moment between requests to avoid rate limiting
            time.sleep(2)
        
        # Print summary
        success_count = sum(1 for r in results if r["status"] == "SUCCESS")
        rate_limited_count = sum(1 for r in results if r["status"] == "RATE LIMITED")
        failed_count = sum(1 for r in results if r["status"] in ["FAILED", "ERROR"])
        partial_count = sum(1 for r in results if r["status"] == "PARTIAL SUCCESS")
        
        print("\n=== SUMMARY ===")
        print(f"Total keys tested: {len(results)}")
        print(f"Successful: {success_count} ({success_count/len(results)*100:.1f}%)")
        print(f"Rate limited: {rate_limited_count} ({rate_limited_count/len(results)*100:.1f}%)")
        print(f"Failed: {failed_count} ({failed_count/len(results)*100:.1f}%)")
        print(f"Partial success: {partial_count} ({partial_count/len(results)*100:.1f}%)")
        
        # Save results to CSV with range information
        filename = f"api_key_check_results_{args.start}-{args.end}.csv"
        save_results_to_csv(results, filename)
        
        print("\nRate-limited keys:")
        rate_limited_keys = [r["key_number"] for r in results if r["status"] == "RATE LIMITED"]
        if rate_limited_keys:
            print(f"Keys: {', '.join(map(str, rate_limited_keys))}")
        else:
            print("None!")
        
        # Return code based on results
        if rate_limited_count + failed_count > 0:
            return 1
        else:
            return 0
    
    sys.exit(ranged_main())