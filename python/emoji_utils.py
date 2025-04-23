#!/usr/bin/env python3

"""
Emoji Utilities for PanicSense NLP Pipeline
Converts emojis to text and preserves exclamation points
"""

import re
import logging

# Dictionary mapping emojis to their textual equivalents
EMOJI_TO_TEXT = {
    # Panic-related emojis
    "😱": "scream face",
    "😭": "crying face",
    "🆘": "sos signal",
    "💔": "broken heart",
    
    # Fear/Anxiety related emojis
    "😨": "fearful face",
    "😰": "anxious face with sweat",
    "😟": "worried face",
    
    # Resilience-related emojis
    "💪": "flexed biceps strength",
    "🙏": "folded hands prayer",
    "🌈": "rainbow hope",
    "🕊️": "dove peace",
    
    # Neutral indicators
    "📍": "location pin",
    "📰": "newspaper",
    
    # Disbelief/sarcasm indicators
    "🤯": "exploding head",
    "🙄": "rolling eyes",
    "😆": "laughing face",
    "😑": "expressionless face",
    
    # Additional common Filipino reaction emojis
    "😂": "laughing with tears",
    "🤣": "rolling on floor laughing",
    "😅": "smiling face with sweat",
    "👀": "looking eyes",
    "👁️": "eye",
    "🔥": "fire",
    "💥": "collision",
    "⚡": "lightning",
    "❤️": "heart",
    "👍": "thumbs up",
    "🤝": "handshake",
}

def preprocess_text(text):
    """
    Preprocess text by converting emojis to their text equivalents
    while preserving exclamation points as panic indicators
    
    Args:
        text (str): The input text to preprocess
        
    Returns:
        str: Preprocessed text with emojis converted to words and exclamation points preserved
    """
    if not text:
        return text
    
    # First, count emojis for statistical purposes
    emoji_count = sum(1 for char in text if ord(char) > 127000)
    
    # Convert emojis to text
    processed_text = text
    for emoji, description in EMOJI_TO_TEXT.items():
        if emoji in processed_text:
            # Add brackets to clearly denote the emoji text
            replacement = f" [{description}] "
            processed_text = processed_text.replace(emoji, replacement)
    
    # Preserve exclamation points (don't normalize them away)
    # This step is important as exclamation points are strong indicators of panic
    
    # Log the preprocessing
    if emoji_count > 0:
        logging.info(f"Converted {emoji_count} emojis to text in: {text[:30]}...")
    
    return processed_text

def preserve_exclamations(text):
    """
    Ensure exclamation points are preserved as they're important panic indicators
    
    Args:
        text (str): Input text that might have exclamation points
        
    Returns:
        str: Text with exclamation points preserved
    """
    # Count consecutive exclamation points as emphasis indicators
    # Replace patterns like !!! with a normalized form that will be recognized by sentiment analysis
    exclamation_pattern = re.compile(r'(!{2,})')
    
    # For !!! or more, add [STRONG EMPHASIS] marker
    if exclamation_pattern.search(text):
        # Replace consecutive exclamations but preserve at least one
        processed_text = exclamation_pattern.sub(r'! [STRONG EMPHASIS]', text)
        return processed_text
    
    return text

def clean_text_preserve_indicators(text):
    """
    Comprehensive text cleaning that preserves important sentiment indicators
    like exclamation points, emojis (converted to text), and capitalization patterns
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned text with sentiment indicators preserved
    """
    if not text:
        return text
    
    # First convert emojis to text equivalents
    processed_text = preprocess_text(text)
    
    # Then ensure exclamation points are preserved
    processed_text = preserve_exclamations(processed_text)
    
    # Preserve ALL CAPS as it's an important indicator of emphasis/panic
    # This is done by noting the ALL CAPS sections rather than normalizing them
    caps_pattern = re.compile(r'\b[A-Z]{2,}\b')
    for caps_word in caps_pattern.findall(processed_text):
        # Mark words that are ALL CAPS but don't change them
        if len(caps_word) >= 3:  # Only mark if at least 3 letters to avoid acronyms
            processed_text = processed_text.replace(caps_word, f"{caps_word} [EMPHASIS]")
    
    return processed_text

if __name__ == "__main__":
    # Test the utility functions
    test_texts = [
        "TULONG!!! May sunog sa tabi ng bahay namin 🔥",
        "Grabe ang lindol 😱 Lahat ng gamit bumagsak!!!",
        "Hahahaha 'TULONG' daw 😂 nakita niyo ba yung video?",
        "Ingat kayo diyan sa area niyo 🙏 stay safe!"
    ]
    
    for text in test_texts:
        print(f"Original: {text}")
        print(f"Processed: {clean_text_preserve_indicators(text)}")
        print("---")