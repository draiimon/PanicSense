#!/usr/bin/env python3

import re
import sys

def main():
    with open('process.py', 'r') as file:
        content = file.read()
    
    # First set of replacements (around line 1543)
    pattern1 = r'"disasterType":\s+csv_disaster\s+if csv_disaster else analysis_result\.get\(\s+"disasterType", "Not Specified"\),'
    replacement1 = '"disasterType":\n            self.extract_disaster_type(text) or\n            csv_disaster or\n            analysis_result.get("disasterType", "Not Specified"),'
    content = re.sub(pattern1, replacement1, content)
    
    pattern2 = r'"location":\s+csv_location if csv_location else\s+analysis_result\.get\("location"\)'
    replacement2 = '"location":\n            self.extract_location(text) or\n            csv_location or\n            analysis_result.get("location")'
    content = re.sub(pattern2, replacement2, content)
    
    with open('process.py', 'w') as file:
        file.write(content)
    
    print("Replacements completed.")

if __name__ == "__main__":
    main()
