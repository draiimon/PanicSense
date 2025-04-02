#!/usr/bin/env python3

# This script creates a modified version of process.py with our changes

import re

def main():
    with open('process.py', 'r') as file:
        content = file.read()
    
    # Modified approach - we'll use string markers with added context to make replacements
    # Modify the first instance of disaster type and location
    content = content.replace(
        """                            "disasterType":
                            csv_disaster
                            if csv_disaster else analysis_result.get(
                                "disasterType", "Not Specified"),
                            "location":
                            csv_location if csv_location else
                            analysis_result.get("location")""",
        
        """                            "disasterType":
                            self.extract_disaster_type(text) or
                            csv_disaster or
                            analysis_result.get("disasterType", "Not Specified"),
                            "location":
                            self.extract_location(text) or
                            csv_location or
                            analysis_result.get("location")"""
    )
    
    # Write modified content to temporary file
    with open('process_temp.py', 'w') as file:
        file.write(content)
    
    print("Created modified version at process_temp.py")

if __name__ == "__main__":
    main()
