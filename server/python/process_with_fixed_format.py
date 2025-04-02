#!/usr/bin/env python3

import re
import sys

def main():
    with open('process.py', 'r') as file:
        content = file.read()
    
    # Fix all instances of disaster type and location extraction
    content = content.replace(
        '"disasterType":\n                            self.extract_disaster_type(text) or\ncsv_disaster or\nanalysis_result.get("disasterType", "Not Specified"),\n"location":\n                            self.extract_location(text) or\ncsv_location or\nanalysis_result.get("location")',
        '"disasterType":\n                            self.extract_disaster_type(text) or\n                            csv_disaster or\n                            analysis_result.get("disasterType", "Not Specified"),\n                            "location":\n                            self.extract_location(text) or\n                            csv_location or\n                            analysis_result.get("location")'
    )
    
    with open('process.py', 'w') as file:
        file.write(content)
    
    print("Formatting fixed in process.py")

if __name__ == "__main__":
    main()
