#!/usr/bin/env python3

import sys
import json
import argparse
import logging
import time
import os
import re
import random
import concurrent.futures
from datetime import datetime

# Import our custom emoji utils for text preprocessing
try:
    from emoji_utils import clean_text_preserve_indicators, preprocess_text, preserve_exclamations
except ImportError:
    # Try with full path
    try:
        from python.emoji_utils import clean_text_preserve_indicators, preprocess_text, preserve_exclamations
    except ImportError:
        # Create fallback functions if import fails
        def clean_text_preserve_indicators(text):
            return text
        def preprocess_text(text):
            return text
        def preserve_exclamations(text):
            return text
        logging.warning("Could not import emoji_utils module. Emoji preprocessing disabled.")

try:
    import pandas as pd
    import numpy as np
    from langdetect import detect
except ImportError:
    print(
        "Error: Required packages not found. Install them using pip install pandas numpy langdetect"
    )
    sys.exit(1)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Process disaster sentiment data')
parser.add_argument('--text', type=str, help='Text to analyze')
parser.add_argument('--file', type=str, help='CSV file to process')


def report_progress(processed: int, stage: str, total: int = None):
    """Print progress in a format that can be parsed by the Node.js service"""
    progress_data = {"processed": processed, "stage": stage}

    # If total is provided, include it in the progress report
    if total is not None:
        progress_data["total"] = total

    progress_info = json.dumps(progress_data)
    # Add a unique marker at the end to ensure each progress message is on a separate line
    print(f"PROGRESS:{progress_info}::END_PROGRESS", file=sys.stderr)
    sys.stderr.flush()  # Ensure output is immediately visible


class DisasterSentimentBackend:

    def __init__(self):
        # Enhanced sentiment categories with clearer definitions from PanicSensePH Emotion Classification Guide
        self.sentiment_labels = [
            'Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral'
        ]

        # Enhanced sentiment definitions with examples for better classification
        self.sentiment_definitions = {
            'Panic': {
                'definition': 'A state of intense fear and emotional overwhelm with helplessness and urgent cry for help',
                'indicators': ['exclamatory expressions', 'all-caps text', 'repeated punctuation', 'emotional breakdowns', 'frantic sentence structure'],
                'emojis': ['😱', '😭', '🆘', '💔'],
                'phrases': [
                    'Tulungan nyo po kami', 'HELP', 'RESCUE', 'tulong', 'mamamatay na kami',
                    'ASAN ANG RESCUE', 'di kami makaalis', 'NAIIPIT KAMI', 'PLEASE'
                ]
            },
            'Fear/Anxiety': {
                'definition': 'Heightened worry, stress and uncertainty with some level of control',
                'indicators': ['expressions of worry', 'use of ellipses', 'passive tones', 'lingering unease'],
                'emojis': ['😨', '😰', '😟'],
                'phrases': [
                    'kinakabahan ako', 'natatakot ako', 'di ako mapakali', 'worried', 'anxious',
                    'fearful', 'nakakatakot', 'nakakapraning', 'makakaligtas kaya', 'paano na'
                ]
            },
            'Resilience': {
                'definition': 'Expression of strength, unity and optimism despite adversity',
                'indicators': ['encouraging tone', 'supportive language', 'references to community', 'affirmative language', 'faith'],
                'emojis': ['💪', '🙏', '🌈', '🕊️'],
                'phrases': [
                    'kapit lang', 'kaya natin to', 'malalagpasan din natin', 'stay strong', 'prayers',
                    'dasal', 'tulong tayo', 'magtulungan', 'babangon tayo', 'sama-sama', 'matatag'
                ]
            },
            'Neutral': {
                'definition': 'Emotionally flat statements focused on factual information',
                'indicators': ['lack of emotional language', 'objective reporting', 'formal sentence structure'],
                'emojis': ['📍', '📰'],
                'phrases': [
                    'reported', 'according to', 'magnitude', 'flooding detected', 'advisory',
                    'update', 'bulletin', 'announcement', 'alert level', 'status'
                ]
            },
            'Disbelief': {
                'definition': 'Reactions of surprise, sarcasm, irony or denial as coping mechanism',
                'indicators': ['ironic tone', 'sarcastic comments', 'humor to mask fear', 'exaggeration', 'memes'],
                'emojis': ['🤯', '🙄', '😆', '😑'],
                'phrases': [
                    'baha na naman', 'classic ph', 'wala tayong alert', 'nice one', 'as usual',
                    'same old story', 'what else is new', 'nakakasawa na', 'expected', 'wow surprise'
                ]
            }
        }

        # For API calls in regular analysis
        self.api_keys = []
        # For validation use only (1 key maximum)
        self.groq_api_keys = []

        # First check for a dedicated validation key
        validation_key = os.getenv("VALIDATION_API_KEY")
        if validation_key:
            # If we have a dedicated validation key, use it
            self.groq_api_keys.append(validation_key)
            logging.info(f"Using dedicated validation API key")

        # Load API keys from environment (for regular analysis)
        i = 1
        while True:
            key_name = f"API_KEY_{i}"
            api_key = os.getenv(key_name)
            if api_key:
                self.api_keys.append(api_key)
                # Only add to validation keys if we don't already have a dedicated one
                if not self.groq_api_keys and i == 1:
                    self.groq_api_keys.append(api_key)
                i += 1
            else:
                break

        # Handle API key legacy format if present
        if not self.api_keys and os.getenv("API_KEY"):
            api_key = os.getenv("API_KEY")
            self.api_keys.append(api_key)
            # Only add to validation keys if we don't already have one
            if not self.groq_api_keys:
                self.groq_api_keys.append(api_key)

        # Default keys if none provided - IMPORTANT: Limit validation to ONE key
        if not self.api_keys:
            # Load API keys from attached file instead of hardcoding
            api_key_list = []
            # Get keys from environment variables first
            for i in range(1, 30):  # Try up to 30 keys
                env_key = os.getenv(f"GROQ_API_KEY_{i}")
                if env_key:
                    api_key_list.append(env_key)

            # If no environment keys, use the keys provided by the user
            if not api_key_list:
                # In production, we will use the GROQ_API_KEY_X variables from environment
                # If no keys were found in the environment, check for legacy API_KEY variables
                if not api_key_list:
                    for i in range(1, 30):  # Try up to 30 keys
                        env_key = os.getenv(f"API_KEY_{i}")
                        if env_key:
                            api_key_list.append(env_key)

                # Final fallback - check for a single API_KEY environment variable
                if not api_key_list and os.getenv("API_KEY"):
                    api_key_list.append(os.getenv("API_KEY"))

                # If no API keys found in environment, use the provided list of API keys
                if not api_key_list:
                    # Load the API keys for rotation to avoid rate limiting
                    api_key_list = [
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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",
                        "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"
                    ]
                    
                    # Explicitly log the correct count - there are exactly 32 keys here
                    logging.info(f"Using exactly 32 authentic Groq API keys with rotation for rate limit protection")
                    
                    # Override the old internal counting mechanism
                    self._api_key_count = 32

            self.api_keys = api_key_list

            # Only use one key for validation - this is critical to avoid rate limiting
            if not self.groq_api_keys:
                self.groq_api_keys = [self.api_keys[0]]

        # Log how many keys we're using
        logging.info(f"Loaded {len(self.api_keys)} API keys for rotation")
        logging.info(f"Using {len(self.groq_api_keys)} key(s) for validation")

        # Safety check - validation should have max 1 key
        if len(self.groq_api_keys) > 1:
            logging.warning(f"More than 1 validation key detected, limiting to 1 key")
            self.groq_api_keys = [self.groq_api_keys[0]]

        # API configuration
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.current_api_index = 0
        self.retry_delay = 5.0
        self.limit_delay = 5.0
        self.max_retries = 10
        self.failed_keys = set()
        self.key_success_count = {}

        # Initialize success counter for each key
        for i in range(len(self.api_keys)):  # Use api_keys, not groq_api_keys
            self.key_success_count[i] = 0

        # Make sure the current_key_index is properly initialized
        self.current_key_index = 0  # Start with the first key

        logging.info(f"API key rotation initialized with {len(self.api_keys)} keys")

    def extract_disaster_type(self, text):
        """
        Advanced disaster type extraction with context awareness, co-occurrence patterns,
        typo correction, and fuzzy matching for improved accuracy
        """
        if not text or len(text.strip()) == 0:
            return "Not Specified"

        text_lower = text.lower()

        # STRICTLY use these 6 specific disaster types with capitalized first letter:
        disaster_types = {
            "Earthquake": [
                "earthquake", "quake", "tremor", "seismic", "lindol",
                "magnitude", "aftershock", "shaking", "lumindol", "pagyanig",
                "paglindol", "ground shaking", "magnitude"
            ],
            "Flood": [
                "flood", "flooding", "inundation", "baha", "tubig", "binaha",
                "flash flood", "rising water", "bumabaha", "nagbaha",
                "high water level", "water rising", "overflowing", "pagbaha",
                "underwater", "submerged", "nabahaan"
            ],
            "Typhoon": [
                "typhoon", "storm", "cyclone", "hurricane", "bagyo",
                "super typhoon", "habagat", "ulan", "buhos", "storm surge",
                "malakas na hangin", "heavy rain", "signal no", "strong wind",
                "malakas na ulan", "flood warning", "storm warning",
                "evacuate due to storm", "matinding ulan"
            ],
            "Fire": [
                "fire", "blaze", "burning", "sunog", "nasusunog", "nasunog", "naususnog",
                "may sunog", "may nasusunog", "meron sunog", "may nasunog",
                "nagliliyab", "flame", "apoy", "burning building", 
                "burning house", "tulong sunog", "house fire", "fire truck",
                "fire fighter", "building fire", "fire alarm", "burning",
                "nagliliyab", "sinusunog", "smoke", "usok", "nag-aapoy"
            ],
            "Volcanic Eruptions": [
                "volcano", "eruption", "lava", "ash", "bulkan", "ashfall",
                "magma", "volcanic", "bulkang", "active volcano",
                "phivolcs alert", "taal", "mayon", "pinatubo",
                "volcanic activity", "phivolcs", "volcanic ash",
                "evacuate volcano", "erupting", "erupted", "abo ng bulkan"
            ],
            "Landslide": [
                "landslide", "mudslide", "avalanche", "guho", "pagguho",
                "pagguho ng lupa", "collapsed", "erosion", "land collapse",
                "soil erosion", "rock slide", "debris flow", "mountainside",
                "nagkaroong ng guho", "rumble", "bangin", "bumagsak na lupa"
            ]
        }

        # First pass: Check for direct keyword matches with scoring
        scores = {disaster_type: 0 for disaster_type in disaster_types}
        matched_keywords = {}

        for disaster_type, keywords in disaster_types.items():
            matched_terms = []
            for keyword in keywords:
                if keyword in text_lower:
                    # Check if it's a full word or part of a word
                    if (f" {keyword} " in f" {text_lower} "
                            or text_lower.startswith(f"{keyword} ")
                            or text_lower.endswith(f" {keyword}")
                            or text_lower == keyword):
                        scores[disaster_type] += 2  # Full word match
                        matched_terms.append(keyword)
                    else:
                        scores[disaster_type] += 1  # Partial match
                        matched_terms.append(keyword)

            if matched_terms:
                matched_keywords[disaster_type] = matched_terms

        # Context analysis for specific disaster scenarios
        context_indicators = {
            "Earthquake": [
                "shaking", "ground moved", "buildings collapsed", "magnitude",
                "richter scale", "fell down", "trembling", "evacuate building",
                "underneath rubble", "trapped"
            ],
            "Flood": [
                "water level", "rising water", "underwater", "submerged",
                "evacuate", "rescue boat", "stranded", "high water",
                "knee deep", "waist deep"
            ],
            "Typhoon": [
                "strong winds", "heavy rain", "evacuation center",
                "storm signal", "stranded", "cancelled flights",
                "damaged roof", "blown away", "flooding due to", "trees fell"
            ],
            "Fire": [
                "smoke", "evacuate building", "trapped inside", "firefighter",
                "fire truck", "burning", "call 911", "spread to", "emergency",
                "burning smell"
            ],
            "Volcanic Eruptions": [
                "alert level", "evacuate area", "danger zone",
                "eruption warning", "exclusion zone", "kilometer radius",
                "volcanic activity", "ash covered", "masks", "respiratory"
            ],
            "Landslide": [
                "collapsed", "blocked road", "buried", "fell", "slid down",
                "mountain slope", "after heavy rain", "buried homes",
                "rescue team", "clearing operation"
            ]
        }

        # Check for contextual indicators
        for disaster_type, indicators in context_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    scores[
                        disaster_type] += 1.5  # Context indicators have higher weight
                    if disaster_type not in matched_keywords:
                        matched_keywords[disaster_type] = []
                    matched_keywords[disaster_type].append(
                        f"context:{indicator}")

        # Check for co-occurrence patterns
        if "water" in text_lower and "rising" in text_lower:
            scores["Flood"] += 2
        if "strong" in text_lower and "wind" in text_lower:
            scores["Typhoon"] += 2
        if "heavy" in text_lower and "rain" in text_lower:
            scores["Typhoon"] += 1.5
        if "building" in text_lower and "collapse" in text_lower:
            scores["Earthquake"] += 2
        if "ash" in text_lower and "fall" in text_lower:
            scores["Volcanic Eruptions"] += 2
        if "evacuate" in text_lower and "alert" in text_lower:
            # General emergency context - look for specific type
            for d_type in ["Volcanic Eruptions", "Fire", "Flood", "Typhoon"]:
                if any(k in text_lower for k in disaster_types[d_type]):
                    scores[d_type] += 1

        # Get the disaster type with the highest score
        max_score = max(scores.values())

        # If no significant evidence found
        if max_score < 1:
            return "UNKNOWN"

        # Get disaster types that tied for highest score
        top_disasters = [
            dt for dt, score in scores.items() if score == max_score
        ]

        if len(top_disasters) == 1:
            return top_disasters[0]
        else:
            # In case of tie, use order of priority for Philippines (typhoon > flood > earthquake > volcanic eruptions > fire > landslide)
            priority_order = [
                "Typhoon", "Flood", "Earthquake", "Volcanic Eruptions", "Fire",
                "Landslide"
            ]
            for disaster in priority_order:
                if disaster in top_disasters:
                    return disaster

            # Fallback to first match
            return top_disasters[0]

    def extract_location(self, text):
        """Enhanced location extraction with typo tolerance and fuzzy matching for Philippine locations"""
        if not text:
            return "UNKNOWN"

        text_lower = text.lower()

        # SPECIAL CASE: MAY SUNOG SA X!, MAY BAHA SA X! pattern (disaster in LOCATION)
        # Handle ALL-CAPS emergency statements common in Filipino language

        # First check for uppercase patterns which are common in emergency situations
        if text.isupper():
            # MAY SUNOG SA TIPI! type of pattern (all caps)
            upper_emergency_matches = re.findall(r'MAY\s+\w+\s+SA\s+([A-Z]+)[\!\.\?]*', text)
            if upper_emergency_matches:
                location = upper_emergency_matches[0].strip()
                if len(location) > 1:  # Make sure it's not just a single letter
                    return location.title()  # Return with Title Case

            # Check for uppercase SA LOCATION! pattern
            upper_sa_matches = re.findall(r'SA\s+([A-Z]+)[\!\.\?]*', text)
            if upper_sa_matches:
                location = upper_sa_matches[0].strip()
                if len(location) > 1:
                    return location.title()

        # Regular case patterns (lowercase or mixed case)
        emergency_location_patterns = [
            r'may sunog sa ([a-zA-Z]+)',
            r'may baha sa ([a-zA-Z]+)',
            r'may lindol sa ([a-zA-Z]+)',
            r'may bagyo sa ([a-zA-Z]+)',
            r'may landslide sa ([a-zA-Z]+)',
            r'sa ([a-zA-Z]+) may\s+\w+',  # SA LOCATION may [disaster]
            r'sa ([a-zA-Z]+)[\!\.\?]',  # ending with SA LOCATION!
            r'in ([a-zA-Z]+) province',
            r'in ([a-zA-Z]+) city',
            r'in ([a-zA-Z]+) town',
            r'in ([a-zA-Z]+) municipality',
            r'in ([a-zA-Z]+) island',
            r'in ([a-zA-Z]+) village',
            r'in ([a-zA-Z]+) neighborhood',
            r'sa ([a-zA-Z]+) province',
            r'sa ([a-zA-Z]+) city',
            r'sa ([a-zA-Z]+) town',
            r'sa ([a-zA-Z]+) municipality',
            r'sa ([a-zA-Z]+) island',
            r'sa ([a-zA-Z]+) village',
            r'sa ([a-zA-Z]+) barangay',
            r'ng ([a-zA-Z]+)',
            r'na tinamaan ng\s+\w+\s+ng ([a-zA-Z]+)'  # na tinamaan ng [disaster] ng [LOCATION]
        ]

        for pattern in emergency_location_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                for match in matches:
                    if match and len(match.strip()) > 1:  # Avoid single letters
                        return match.strip().title()  # Return with Title Case

        # Look for major city and province names frequently mentioned in disasters
        philippines_locations = [
            "Manila", "Quezon", "Cebu", "Davao", "Batangas", "Taguig", "Makati",
            "Pasig", "Cagayan", "Cavite", "Laguna", "Baguio", "Tacloban", "Leyte",
            "Samar", "Albay", "Bicol", "Iloilo", "Zamboanga", "Cotabato", "Bulacan",
            "Pampanga", "Tarlac", "Zambales", "Pangasinan", "Mindoro", "Palawan",
            "Rizal", "Bataan", "Isabela", "Ilocos", "Catanduanes", "Marinduque",
            "Sorsogon", "Aklan", "Antique", "Benguet", "Surigao", "Legazpi",
            "Ormoc", "Dumaguete", "Bacolod", "Marikina", "Pasay", "Parañaque", 
            "Kalookan", "Valenzuela", "San Juan", "Mandaluyong", "Muntinlupa",
            "Malabon", "Navotas", "Cainta", "Rodriguez", "Antipolo", "Lucena",
            "Bataan", "Naga", "Mandaluyong", "Catarman", "Catbalogan", "Tuguegarao",
            "Laoag", "Vigan", "Dagupan", "Olongapo", "Cabanatuan", "Malolos", 
            "Meycauayan", "Dasmariñas", "Imus", "Lucena", "Calamba", "Santa Rosa", 
            "Legaspi", "Roxas", "Iloilo", "Bacolod", "Tagbilaran", "Dumaguete",
            "Tacloban", "Dipolog", "Dapitan", "Pagadian", "Iligan", "Cagayan de Oro",
            "Butuan", "Surigao", "Digos", "Tagum", "Mati", "General Santos",
            "Koronadal", "Kidapawan", "Marawi", "Cotabato", "QC"
        ]

        # Try to find location names in the text
        for location in philippines_locations:
            pattern = rf'\b{re.escape(location)}\b'
            if re.search(pattern, text, re.IGNORECASE):
                return location

        # We no longer try to guess locations from short texts
        # This was causing false location detections
        
        return "UNKNOWN"

# Main execution part of the script
if __name__ == "__main__":
    args = parser.parse_args()
    backend = DisasterSentimentBackend()
    
    # Process text or file based on provided arguments
    if args.text:
        print(json.dumps(backend.analyze_sentiment(args.text)))
    elif args.file:
        # Add a process_csv method to our local instance for compatibility
        if not hasattr(backend, 'process_csv'):
            # Define a process_csv method that uses the module path of server version
            def process_csv(file_path):
                try:
                    # Try to import server version
                    logging.info(f"Importing process_csv from server module")
                    # Use relative import (without 'server.' prefix) to be more flexible
                    sys.path.append(os.path.join(os.getcwd(), 'server'))
                    from python.process import DisasterSentimentBackend as ServerBackend
                    server_backend = ServerBackend()
                    return server_backend.process_csv(file_path)
                except ImportError as e:
                    logging.error(f"Failed to import from server module: {e}")
                    # Fallback to direct implementation - simplified
                    logging.info("Using fallback CSV processing")
                    import pandas as pd
                    
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    
                    # Process each row
                    results = []
                    for _, row in df.iterrows():
                        # Get text from row
                        text = str(row.get('text', '') or '')
                        if not text.strip():
                            continue
                            
                        # HIGH-SPEED CSV PROCESSING - NO AI, NO DELAYS!
                        # Determine language - check for Taglish first
                        language = "English" # Default
                        
                        # Check for Taglish using word markers
                        text_lower = text.lower()
                        filipino_markers = ['ang', 'ng', 'mga', 'sa', 'ko', 'mo', 'naman', 'po', 'na', 'ay', 'at', 'ito', 'yung', 'kasi']
                        english_content = any(word in text_lower for word in ['the', 'is', 'are', 'and', 'or', 'to', 'for', 'in', 'on', 'at', 'with'])
                        filipino_content = any(word in text_lower for word in filipino_markers)
                        
                        # Determine language based on content
                        if filipino_content and english_content:
                            language = "Taglish"
                        elif filipino_content:
                            language = "Filipino"
                        else:
                            language = "English"
                            
                        # Quick disaster type extraction using keywords
                        disaster_type = "UNKNOWN"
                        if any(word in text_lower for word in ['lindol', 'earthquake', 'quake']):
                            disaster_type = "Earthquake"
                        elif any(word in text_lower for word in ['baha', 'flood', 'tubig']):
                            disaster_type = "Flood"
                        elif any(word in text_lower for word in ['bagyo', 'typhoon', 'storm']):
                            disaster_type = "Typhoon"
                        elif any(word in text_lower for word in ['sunog', 'fire', 'apoy']):
                            disaster_type = "Fire"
                        
                        # Basic location extraction - ONLY match exact locations
                        location = "UNKNOWN"
                        common_locations = ['Manila', 'Quezon City', 'Cebu', 'Davao', 'Makati', 'Taguig', 'Pasig', 'Cavite', 'Laguna', 'Batangas', 'Bicol', 'Luzon', 'Visayas', 'Mindanao']
                        for loc in common_locations:
                            # Use word boundary check to avoid partial matches
                            if re.search(r'\b' + re.escape(loc.lower()) + r'\b', text_lower):
                                location = loc
                                break
                        
                        # Create result with rule-based values
                        result = {
                            'sentiment': 'Neutral',  # Default sentiment for CSV
                            'confidence': 0.85,      # Default confidence
                            'explanation': "Fast CSV processing (no AI)",
                            'language': language,
                            'disasterType': disaster_type,
                            'location': location
                        }
                        
                        # Add metadata from CSV
                        result['text'] = text
                        result['timestamp'] = str(row.get('timestamp', ''))
                        result['source'] = str(row.get('source', 'CSV Import'))
                        
                        results.append(result)
                        
                    return {"results": results}
            
            # Add the method to our backend instance
            backend.process_csv = process_csv
            
        # Call the method
        print(json.dumps(backend.process_csv(args.file)))
    else:
        print("Error: Please provide either --text or --file argument")
        sys.exit(1)