#!/usr/bin/env python3

from datetime import datetime
import sys

def test_iso8601_parsing():
    """Test parsing various ISO 8601 formatted timestamps"""
    print("Testing ISO 8601 timestamp parsing in Python\n")
    
    # Test cases
    timestamps = [
        "2019-12-25T04:27:04.000Z",       # From dataset
        "2023-05-01T12:30:45.123Z",       # Another example
        datetime.now().isoformat() + "Z"   # Current time with Z suffix
    ]
    
    for i, ts in enumerate(timestamps):
        print(f"Test case {i+1}: {ts}")
        
        # Method 1: Replace Z with +00:00 (works in Python 3.7+)
        try:
            date1 = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            print(f"  ✅ Method 1 (replace Z): {date1}")
        except Exception as e:
            print(f"  ❌ Method 1 failed: {e}")
        
        # Method 2: Direct parsing (may require Python 3.11+)
        try:
            date2 = datetime.fromisoformat(ts)
            print(f"  ✅ Method 2 (direct): {date2}")
        except Exception as e:
            print(f"  ❌ Method 2 failed: {e}")
            
        # Method 3: Fall back to strptime
        try:
            date3 = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
            print(f"  ✅ Method 3 (strptime): {date3}")
        except Exception as e:
            print(f"  ❌ Method 3 failed: {e}")
        
        print("-" * 50)
    
    # Test CSV row parsing mechanism
    print("\nTesting CSV row timestamp extraction (as in process.py):")
    
    # Mock CSV row
    row = {
        "date": "2019-12-25T04:27:04.000Z",
        "text": "Test text",
        "source": "Facebook"
    }
    
    # Get timestamp with fallback (similar to the process.py code)
    timestamp_col = "date"
    timestamp = str(row.get(timestamp_col, datetime.now().isoformat()))
    print(f"Extracted timestamp: {timestamp}")
    
    # Test if it converts back to datetime
    try:
        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        print(f"Converted back to datetime: {date}")
        print("✅ CSV timestamp extraction works")
    except Exception as e:
        print(f"❌ CSV timestamp extraction failed: {e}")

if __name__ == "__main__":
    test_iso8601_parsing()