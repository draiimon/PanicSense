#!/usr/bin/env python3
import sys
import re

def extract_methods(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the "trainModelWithFeedback" method
    train_method_match = re.search(r'public async trainModelWithFeedback\(.*?\)\s*{.*?^  }', 
                                 content, re.DOTALL | re.MULTILINE)
    
    if train_method_match:
        train_method = train_method_match.group(0)
        print("TRAIN_METHOD_START")
        print(train_method)
        print("TRAIN_METHOD_END")
    else:
        print("Could not find trainModelWithFeedback method")
    
    # Find the "processCSV" method
    process_method_match = re.search(r'public async processCSV\(.*?\)\s*{.*?^  }', 
                                   content, re.DOTALL | re.MULTILINE)
    
    if process_method_match:
        process_method = process_method_match.group(0)
        print("PROCESS_METHOD_START")
        print(process_method)
        print("PROCESS_METHOD_END")
    else:
        print("Could not find processCSV method")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <typescript_file>")
        sys.exit(1)
    
    extract_methods(sys.argv[1])