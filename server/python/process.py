#!/usr/bin/env python3
import sys
import pandas as pd

def process_csv_file(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin1', error_bad_lines=False)
        print("Read CSV successfully")
        results = []
        for index, row in df.iterrows():
            text = str(row.iloc[0])
            if text.strip():
                results.append({'text': text})
        return {'results': results}
    except Exception as e:
        print(f"Failed to process CSV: {str(e)}")
        raise

def main():
    if len(sys.argv) < 2:
        print("Error: No file specified")
        sys.exit(1)
    try:
        result = process_csv_file(sys.argv[1])
        print(result)
        sys.stdout.flush()
    except Exception as e:
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()