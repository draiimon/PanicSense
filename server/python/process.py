#!/usr/bin/env python3
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from langdetect import detect
import logging
import re
import os
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GroqPredictor:
    def __init__(self):
        self.api_keys = []

        # Look for API keys in environment variables (API_KEY_1, API_KEY_2, etc.)
        i = 1
        while True:
            key_name = f"API_KEY_{i}"
            api_key = os.getenv(key_name)
            if api_key:
                self.api_keys.append(api_key)
                i += 1
            else:
                break

        # Fallback to provided keys if no environment variables found
        if not self.api_keys:
            self.api_keys = [
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

        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.current_key_index = 0
        self.max_retries = 3
        self.retry_delay = 1
        self.limit_delay = 0.5

    def get_next_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logging.info(f"Switching to API key {self.current_key_index + 1}/{len(self.api_keys)}")
        return self.api_keys[self.current_key_index]

    def analyze_with_groq(self, text, retry_count=0):
        try:
            headers = {
                "Authorization": f"Bearer {self.api_keys[self.current_key_index]}",
                "Content-Type": "application/json"
            }

            prompt = f"""Analyze the following text for disaster sentiment.
Be very specific about disaster details, location, and emotions.

Text: {text}

Provide a strict JSON response in this format:
{{
    "disaster_type": string or null,
    "disaster_confidence": float between 0 and 1,
    "disaster_explanation": string with details about why this disaster was detected,
    "location": string or null,
    "location_type": "city" or "region" or "province" or null,
    "sentiment": "Panic" or "Fear/Anxiety" or "Disbelief" or "Resilience" or "Neutral",
    "confidence": float between 0 and 1,
    "explanation": string giving detailed reasoning for the sentiment
}}

Consider both English and Filipino/Tagalog disaster terminology and expressions."""

            payload = {
                "model": "mixtral-8x7b-32768",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a disaster analysis AI specializing in Filipino and English text analysis. Focus on precise disaster detection, location identification, and emotional sentiment analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000,
                "response_format": {"type": "json_object"}
            }

            logging.info(f"Sending request to Groq API (attempt {retry_count + 1}/{self.max_retries})")
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                try:
                    analysis = json.loads(result["choices"][0]["message"]["content"])
                    logging.info("Successfully received and parsed Groq API response")
                    return analysis
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse Groq response as JSON: {e}")
                    return {"error": "Failed to parse Groq response as JSON"}
            else:
                logging.error("No valid response content from Groq API")
                return {"error": "No valid response from Groq API"}

        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response and e.response.status_code == 429:
                logging.warning(f"Rate limit hit, switching API key (attempt {retry_count + 1})")
                time.sleep(self.limit_delay)
                if retry_count < self.max_retries:
                    self.get_next_api_key()
                    return self.analyze_with_groq(text, retry_count + 1)
            else:
                logging.error(f"API request error: {str(e)}")
                time.sleep(self.retry_delay)
                if retry_count < self.max_retries:
                    return self.analyze_with_groq(text, retry_count + 1)

            return {"error": f"Failed to get response from Groq API after {self.max_retries} retries"}

class DisasterAnalyzer:
    def __init__(self):
        self.groq = GroqPredictor()
        # Keep the existing pattern matching as fallback
        self.patterns = {
            "Earthquake": {
                "primary": [
                    "earthquake", "quake", "tremor", "seismic", "magnitude",
                    "lindol", "linog", "yanig", "alog", "umaalog", "yumanig",
                    "paglindol", "pagyanig", "aftershock", "nayayanig"
                ],
                "related": [
                    "ground shaking", "earth movement", "fault", "epicenter",
                    "lumindol", "yumayaning", "nililindog", "pag-alog",
                    "pag-uga", "paglindol", "lindol na malakas"
                ],
                "impact": [
                    "gumuho", "gumuguho", "bumagsak", "tumumba", "nasira",
                    "nawasak", "collapsed", "damaged", "napinsala", "bitak",
                    "crack", "sumira", "nasiraan", "tumatagilid"
                ],
                "weight": 2.5
            },
            "Flood": {
                "primary": [
                    "flood", "flooding", "baha", "bumabaha", "pagbaha",
                    "binabaha", "tubig", "overflow", "rising water"
                ],
                "related": [
                    "water level", "flash flood", "inundation", "submerged",
                    "babaha", "tumaas ang tubig", "tumataas na tubig",
                    "pag-apaw", "umapaw", "lumagpas", "tumatabon"
                ],
                "impact": [
                    "stranded", "trapped", "submerged", "underwater", "lubog",
                    "nalubog", "hindi madaanan", "hirap dumaan", "napinsala",
                    "nasira", "naipit", "rescue", "naiipit", "nanganganib"
                ],
                "weight": 2.3
            },
            "Typhoon": {
                "primary": [
                    "typhoon", "storm", "bagyo", "cyclone", "hurricane",
                    "tropical storm", "tropical depression", "tropical cyclone"
                ],
                "related": [
                    "signal no", "storm signal", "heavy rain", "strong winds",
                    "malakas na ulan", "malakas na hangin", "hanging habagat",
                    "unos", "daluyong", "pag-ulan", "bumagyo", "bagyong"
                ],
                "impact": [
                    "landslide", "flooding", "storm surge", "evacuation",
                    "damage", "nasira", "nagiba", "natumba", "nawalan ng kuryente",
                    "walang ilaw", "brownout", "blackout", "nagbaha"
                ],
                "weight": 2.4
            }
        }

        # Emergency and intensity indicators
        self.emergency_words = {
            "en": ["emergency", "urgent", "immediate", "help", "rescue", "evacuate", "danger"],
            "tl": ["tulong", "saklolo", "agad", "emergency", "delikado", "panganib", "ligtas"]
        }

        # Location patterns for Philippines
        self.locations = {
            "regions": {
                "NCR": ["metro manila", "ncr", "kamaynilaan", "kalakhang maynila"],
                "CAR": ["cordillera", "car", "mountain province", "benguet"],
                "CALABARZON": ["region 4a", "calabarzon", "southern tagalog"],
                "MIMAROPA": ["region 4b", "mimaropa", "southwestern tagalog"],
                "Bicol": ["region 5", "bicol", "bicolandia", "albay", "sorsogon"],
                "Ilocos": ["region 1", "ilocos", "ilocandia", "pangasinan"]
            },
            "cities": {
                "Manila": ["manila", "maynila", "city of manila"],
                "Quezon City": ["quezon city", "qc", "kyusi", "quezon"],
                "Makati": ["makati", "makati city", "makatí"],
                "Cebu": ["cebu", "cebu city", "sugbo", "cebu city"],
                "Davao": ["davao", "davao city", "dabaw"]
            }
        }

    def tokenize_text(self, text):
        try:
            from sacremoses import MosesTokenizer
            return MosesTokenizer().tokenize(text.lower())
        except ImportError:
            return text.lower().split()

    def detect_disaster(self, text, tokens):
        text_lower = text.lower()
        scores = {}
        explanations = {}

        for disaster_type, patterns in self.patterns.items():
            score = 0
            evidence = []

            # Check primary indicators (highest weight)
            primary_matches = [word for word in patterns["primary"] if word in text_lower]
            if primary_matches:
                score += len(primary_matches) * patterns["weight"] * 1.5
                evidence.extend([f"Primary indicator: {match}" for match in primary_matches])

            # Check related terms (medium weight)
            related_matches = [phrase for phrase in patterns["related"] if phrase in text_lower]
            if related_matches:
                score += len(related_matches) * patterns["weight"]
                evidence.extend([f"Related term: {match}" for match in related_matches])

            # Check impact descriptions (supporting weight)
            impact_matches = [word for word in patterns["impact"] if word in text_lower]
            if impact_matches:
                score += len(impact_matches) * patterns["weight"] * 0.75
                evidence.extend([f"Impact indicator: {match}" for match in impact_matches])

            if score > 0:
                scores[disaster_type] = score
                explanations[disaster_type] = evidence

        if not scores:
            return None

        # Get the highest scoring disaster
        top_disaster = max(scores.items(), key=lambda x: x[1])
        confidence = min(top_disaster[1] / 10, 1.0)

        return {
            "type": top_disaster[0],
            "confidence": confidence,
            "evidence": explanations[top_disaster[0]]
        }

    def detect_location(self, text):
        text_lower = text.lower()

        # First check for regions
        for region, patterns in self.locations["regions"].items():
            if any(pattern in text_lower for pattern in patterns):
                return {"name": region, "type": "region"}

        # Then check for cities
        for city, patterns in self.locations["cities"].items():
            if any(pattern in text_lower for pattern in patterns):
                return {"name": city, "type": "city"}

        return None

    def analyze_sentiment(self, text, disaster_info=None):
        text_lower = text.lower()
        tokens = self.tokenize_text(text)

        # Emotion patterns with multilingual support
        emotions = {
            "Panic": {
                "en": ["help", "emergency", "sos", "trapped", "save", "urgent"],
                "tl": ["tulong", "saklolo", "tulungan", "naipit", "emergency", "agad"],
                "weight": 2.5
            },
            "Fear/Anxiety": {
                "en": ["scared", "afraid", "worried", "fear", "scary", "terrified"],
                "tl": ["takot", "natatakot", "kinakabahan", "kaba", "nerbyos", "balisa"],
                "weight": 2.0
            },
            "Disbelief": {
                "en": ["unbelievable", "cannot believe", "impossible", "shocked"],
                "tl": ["hindi kapani-paniwala", "di makapaniwala", "imposible", "nagulat"],
                "weight": 1.5
            },
            "Resilience": {
                "en": ["strong", "survive", "help", "together", "brave", "hope"],
                "tl": ["malakas", "matibay", "tulong", "sama-sama", "matapang", "pag-asa"],
                "weight": 1.8
            }
        }

        # Calculate emotion scores
        scores = {emotion: 0 for emotion in emotions.keys()}
        evidence = {emotion: [] for emotion in emotions.keys()}

        # Detect language
        lang = detect(text)
        lang_key = "tl" if lang in ["tl", "fil"] else "en"

        # Score each emotion
        for emotion, patterns in emotions.items():
            # Check language-specific patterns
            matches = [word for word in patterns[lang_key] if word in text_lower]
            if matches:
                score = len(matches) * patterns["weight"]
                scores[emotion] = score
                evidence[emotion].extend([f"Found {lang_key} term: {match}" for match in matches])

            # Apply disaster context boost
            if disaster_info and emotion in ["Panic", "Fear/Anxiety"]:
                scores[emotion] *= 1.5
                evidence[emotion].append(f"Boosted due to disaster context: {disaster_info['type']}")

        # Find dominant emotion
        if any(scores.values()):
            dominant = max(scores.items(), key=lambda x: x[1])
            confidence = min(dominant[1] / 10, 1.0)
            return {
                "emotion": dominant[0],
                "confidence": confidence,
                "evidence": evidence[dominant[0]]
            }

        return {
            "emotion": "Neutral",
            "confidence": 0.8,
            "evidence": ["No strong emotions detected"]
        }

    def analyze_with_patterns(self, text):
        tokens = self.tokenize_text(text)
        disaster_info = self.detect_disaster(text, tokens)
        location_info = self.detect_location(text)
        sentiment_info = self.analyze_sentiment(text, disaster_info)

        return {
            "sentiment": sentiment_info["emotion"],
            "confidence": sentiment_info["confidence"],
            "explanation": ", ".join(sentiment_info["evidence"]),
            "disaster_type": disaster_info["type"] if disaster_info else None,
            "disaster_confidence": disaster_info["confidence"] if disaster_info else 0,
            "disaster_explanation": ", ".join(disaster_info["evidence"]) if disaster_info else "No disaster detected",
            "location": location_info["name"] if location_info else None,
            "location_type": location_info["type"] if location_info else None
        }

    def analyze_text(self, text):
        """
        Analyze text using Groq API first, fallback to pattern matching if needed
        """
        # First try Groq API
        logging.info("Attempting analysis with Groq API")
        groq_analysis = self.groq.analyze_with_groq(text)

        if "error" not in groq_analysis:
            logging.info("Successfully analyzed with Groq API")
            return groq_analysis
        else:
            logging.warning(f"Falling back to pattern matching. Groq error: {groq_analysis['error']}")
            return self.analyze_with_patterns(text)

def process_text(text):
    try:
        analyzer = DisasterAnalyzer()
        analysis = analyzer.analyze_text(text)

        return {
            "text": text,
            "language": detect(text),
            "sentiment": analysis.get("sentiment", "Neutral"),
            "confidence": analysis.get("confidence", 0.5),
            "explanation": analysis.get("explanation", "Analysis completed"),
            "disasterType": analysis.get("disaster_type"),
            "disasterConfidence": analysis.get("disaster_confidence", 0),
            "disasterExplanation": analysis.get("disaster_explanation", "No disaster detected"),
            "location": analysis.get("location"),
            "locationType": analysis.get("location_type"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}")
        return {
            "error": str(e),
            "text": text,
            "timestamp": datetime.now().isoformat()
        }

def process_file(filepath):
    try:
        df = pd.read_csv(filepath)
        total_records = len(df)
        results = []

        for idx, row in df.iterrows():
            text = str(row.get('text', row.get('content', '')))
            result = process_text(text)
            results.append(result)

            if idx % 10 == 0:
                progress = {
                    "processed": idx + 1,
                    "stage": f"Processing record {idx + 1} of {total_records}"
                }
                print(f"PROGRESS:{json.dumps(progress)}")
                sys.stdout.flush()

        return {"results": results}
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text for sentiment analysis')
    parser.add_argument('--file', help='Path to the CSV file to analyze')
    parser.add_argument('--text', help='Text to analyze for sentiment')
    args = parser.parse_args()

    try:
        if args.text:
            result = process_text(args.text)
            print(json.dumps(result))
        elif args.file:
            results = process_file(args.file)
            print(json.dumps(results))
        else:
            print(json.dumps({"error": "No input provided"}))
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(json.dumps({"error": str(e)}))