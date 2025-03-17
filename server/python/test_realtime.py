#!/usr/bin/env python3

import sys
import argparse
from pattern_analyzer import analyzer
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_text():
    """Interactive sentiment analysis"""
    print("\n🤖 Disaster Sentiment Analyzer (Type 'exit' to quit)")
    print("📝 Enter text in English or Filipino to analyze:")
    print("💡 Example: 'Help! There's a flood in our area!' or 'Tulong! Bumabaha sa amin!'")
    print("\nType your message and press Enter ⌨️\n")

    while True:
        try:
            text = input("Text to analyze > ").strip()

            # Check for exit command
            if text.lower() in ['exit', 'quit', 'tapos', 'labas']:
                print("\n👋 Salamat sa paggamit! Thank you for using the analyzer!")
                break

            if not text:
                print("⚠️  Please enter some text to analyze!")
                continue

            # Analyze and display results
            display_analysis(text)

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting analyzer...")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def display_analysis(text):
    """Display sentiment analysis results"""
    try:
        # Analyze sentiment
        result = analyzer.analyze_sentiment(text)

        # Get emoji for sentiment
        emoji = {
            'Panic': '😨',
            'Fear/Anxiety': '😰', 
            'Disbelief': '😯',
            'Resilience': '💪',
            'Neutral': '😐'
        }.get(result['sentiment'], '🤔')

        # Print results with nice formatting
        print("\n📊 Analysis Results / Resulta ng Pagsusuri:")
        print("─" * 50)
        print(f"🎯 Sentiment / Damdamin: {emoji} {result['sentiment']}")
        print(f"📈 Confidence / Tiwala: {result['confidence']:.1%}")
        print(f"💭 Explanation / Paliwanag: {result['explanation']}")
        print(f"🗣️ Language / Wika: {result['language']}")
        print("─" * 50 + "\n")

    except Exception as e:
        print(f"❌ Analysis error / May error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive disaster-related sentiment analyzer"
    )
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
        print(f"❌ Fatal error / Malubhang error: {str(e)}")
        sys.exit(1)