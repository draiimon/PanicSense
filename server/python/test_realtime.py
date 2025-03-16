#!/usr/bin/env python3

import sys
import argparse
from sentiment_analyzer import analyzer
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_text():
    """Interactive sentiment analysis"""
    if not sys.stdin.isatty():
        print("Error: This script requires an interactive terminal")
        return

    print("\nDisaster Sentiment Analyzer (Type 'exit' to quit)")
    print("Enter text to analyze (Press Enter after each input):")

    while True:
        try:
            # Get input from user
            text = input("\nText > ").strip()

            # Check for exit command
            if text.lower() in ['exit', 'quit']:
                break

            if not text:
                continue

            display_analysis(text)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def display_analysis(text):
    """Display sentiment analysis results"""
    try:
        # Analyze sentiment
        result = analyzer.analyze_sentiment(text)

        # Print results
        print("\nAnalysis Results:")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Explanation: {result['explanation']}")
        print(f"Language: {result['language']}")

    except Exception as e:
        print(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze disaster-related sentiment in text")
    parser.add_argument("--text", type=str, help="Text to analyze (for non-interactive mode)")

    args = parser.parse_args()

    try:
        if args.text:
            # Non-interactive mode with command line argument
            display_analysis(args.text)
        else:
            # Interactive mode
            analyze_text()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)