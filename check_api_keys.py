#!/usr/bin/env python3

import requests
import time
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# All 45 API keys from server/python/process.py
api_keys = [
    # Key #1-10
    "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",  # Key #1
    "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",  # Key #2
    "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",  # Key #3
    "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",  # Key #4
    "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",  # Key #5
    "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",  # Key #6
    "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",  # Key #7
    "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",  # Key #8
    "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",  # Key #9
    "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",  # Key #10
    
    # Key #11-20
    "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",  # Key #11
    "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",  # Key #12
    "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",  # Key #13
    "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",  # Key #14
    "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",  # Key #15
    "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",  # Key #16
    "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",  # Key #17
    "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",  # Key #18
    "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",  # Key #19
    "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",  # Key #20
    
    # Key #21-30
    "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",  # Key #21
    "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",  # Key #22
    "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",  # Key #23
    "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",  # Key #24
    "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",  # Key #25
    "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",  # Key #26
    "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",  # Key #27
    "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",  # Key #28
    "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",  # Key #29
    "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",  # Key #30
    
    # Key #31-40
    "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",  # Key #31
    "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",  # Key #32
    "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",  # Key #33
    "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",  # Key #34
    "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",  # Key #35
    "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",  # Key #36
    "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",  # Key #37
    "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",  # Key #38
    "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",  # Key #39
    "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",  # Key #40
    
    # Key #41-45
    "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",  # Key #41
    "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",  # Key #42
    "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",  # Key #43
    "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",  # Key #44
    "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",  # Key #45
]

def check_rate_limit(api_key, key_number):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    masked_key = api_key[:10] + "***" if len(api_key) > 10 else "***"
    logging.info(f"Testing key #{key_number}: {masked_key}")
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello"}
                ],
                "temperature": 0.2,
                "max_tokens": 10
            },
            timeout=30
        )
        
        if response.status_code == 200:
            logging.info(f"✅ Key #{key_number} SUCCESS: {masked_key}")
            return {"status": "success", "key_number": key_number}
        elif response.status_code == 429:
            error_msg = response.text
            logging.warning(f"❌ Key #{key_number} RATE LIMITED: {masked_key}")
            logging.warning(f"Response: {error_msg}")
            return {"status": "rate_limited", "key_number": key_number, "error": error_msg}
        else:
            error_msg = response.text
            logging.error(f"❌ Key #{key_number} ERROR (Status {response.status_code}): {masked_key}")
            logging.error(f"Response: {error_msg}")
            return {"status": "error", "key_number": key_number, "error": error_msg, "status_code": response.status_code}
    except Exception as e:
        error_msg = str(e)
        logging.error(f"❌ Key #{key_number} EXCEPTION: {masked_key} - {error_msg}")
        return {"status": "exception", "key_number": key_number, "error": error_msg}

def save_results_to_file(results, filename="api_key_check_results.json"):
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save results to file: {str(e)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check Groq API key rate limits')
    parser.add_argument('--start', type=int, default=1, help='Start checking from this key number (1-indexed)')
    parser.add_argument('--end', type=int, default=None, help='End checking at this key number (1-indexed, inclusive)')
    parser.add_argument('--specific', type=str, help='Comma-separated list of specific key numbers to check (e.g., "37,38,39")')
    parser.add_argument('--problematic-only', action='store_true', help='Only check previously problematic keys (37, 38, 39)')
    args = parser.parse_args()
    
    # Define the previously problematic keys
    problem_keys = [37, 38, 39]
    
    # Determine which keys to check
    keys_to_check = []
    if args.problematic_only:
        logging.info("Only checking previously problematic keys (37, 38, 39)")
        keys_to_check = [(i-1, i) for i in problem_keys]  # (index, key_number) pairs
    elif args.specific:
        specific_keys = [int(k.strip()) for k in args.specific.split(',')]
        logging.info(f"Only checking specific keys: {specific_keys}")
        keys_to_check = [(i-1, i) for i in specific_keys if 1 <= i <= len(api_keys)]
    else:
        end = args.end or len(api_keys)
        logging.info(f"Checking keys from {args.start} to {end}")
        keys_to_check = [(i-1, i) for i in range(args.start, end+1) if 1 <= i <= len(api_keys)]
    
    logging.info("Starting API key rate limit check")
    logging.info(f"Testing {len(keys_to_check)} keys")
    
    results = {
        "working": [],
        "rate_limited": [],
        "error": [],
        "exception": [],
        "details": []
    }
    
    for i, key_number in keys_to_check:
        # Skip if index is out of range
        if i < 0 or i >= len(api_keys):
            logging.warning(f"Key #{key_number} is out of range, skipping")
            continue
            
        key = api_keys[i]
        result = check_rate_limit(key, key_number)
        results["details"].append(result)
        
        if result["status"] == "success":
            results["working"].append(key_number)
        elif result["status"] == "rate_limited":
            results["rate_limited"].append(key_number)
        elif result["status"] == "error":
            results["error"].append(key_number)
        elif result["status"] == "exception":
            results["exception"].append(key_number)
        
        # Sleep to avoid triggering rate limits unnecessarily
        time.sleep(1)
    
    # Save results to file
    save_results_to_file(results)
    
    # Print summary
    logging.info("----- SUMMARY -----")
    logging.info(f"Working keys: {len(results['working'])} - {results['working']}")
    logging.info(f"Rate limited keys: {len(results['rate_limited'])} - {results['rate_limited']}")
    logging.info(f"Error keys: {len(results['error'])} - {results['error']}")
    logging.info(f"Exception keys: {len(results['exception'])} - {results['exception']}")
    
    # Special note about previously problematic keys that were checked
    checked_problem_keys = set(key_number for _, key_number in keys_to_check).intersection(problem_keys)
    for key_num in checked_problem_keys:
        if key_num in results["rate_limited"]:
            logging.warning(f"⚠️ Key #{key_num} - Previously reported problematic key is RATE LIMITED")
        elif key_num in results["error"] or key_num in results["exception"]:
            logging.warning(f"⚠️ Key #{key_num} - Previously reported problematic key has ERROR/EXCEPTION")
        else:
            logging.info(f"✓ Key #{key_num} - Previously reported problematic key is now WORKING")

if __name__ == "__main__":
    main()