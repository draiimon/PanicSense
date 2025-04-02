#!/usr/bin/env python3

# This script makes very specific replacements to process.py
# by targeting exact line numbers and content

def main():
    file_path = 'process.py'
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # First instance (around line 1543-1549)
    if len(lines) >= 1550:
        lines[1543] = '                            "disasterType":
'
        lines[1544] = '                            self.extract_disaster_type(text) or
'
        lines[1545] = '                            csv_disaster or analysis_result.get(
'
        lines[1546] = '                                "disasterType", "Not Specified"),
'
        lines[1547] = '                            "location":
'
        lines[1548] = '                            self.extract_location(text) or
'
        lines[1549] = '                            csv_location or analysis_result.get("location")
'
        
        # Second instance (around line 1756-1762)
        if len(lines) >= 1763:
            lines[1756] = '                            "disasterType":
'
            lines[1757] = '                            self.extract_disaster_type(text) or
'
            lines[1758] = '                            csv_disaster or analysis_result.get(
'
            lines[1759] = '                                "disasterType", "Not Specified"),
'
            lines[1760] = '                            "location":
'
            lines[1761] = '                            self.extract_location(text) or
'
            lines[1762] = '                            csv_location or analysis_result.get("location")
'
    
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    print("Made precise replacements to process.py by line number")

if __name__ == "__main__":
    main()
