#!/usr/bin/env python3
"""
Disaster News Analysis Script
This script analyzes news articles to detect and classify disasters using OpenAI API.
It helps to more accurately categorize disaster types beyond simple keyword matching.
"""

import os
import sys
import json
import time
from datetime import datetime
from openai import OpenAI

# Initialize OpenAI client with API key from environment variable
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Disaster types we're interested in classifying
DISASTER_TYPES = [
    "typhoon",
    "earthquake",
    "flood",
    "fire",
    "landslide",
    "volcanic eruption",
    "tsunami",
    "drought",
    "extreme heat",
    "storm surge",
]

class DisasterAnalyzer:
    def __init__(self, model="gpt-4o-mini"):
        """Initialize the disaster analyzer with the specified OpenAI model."""
        self.model = model
        self.cache = {}  # Simple cache to avoid duplicate API calls
        self.last_call_time = 0  # For rate limiting
        self.min_interval = 0.5  # Minimum time between API calls (seconds)
    
    def _rate_limit(self):
        """Simple rate limiting to avoid hitting OpenAI API limits."""
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        self.last_call_time = time.time()
    
    def analyze_text(self, title, content):
        """
        Analyze news article to determine if it's disaster-related and classify it.
        
        Args:
            title (str): The news article title
            content (str): The news article content
            
        Returns:
            dict: Analysis results with disaster type, location, and severity
        """
        # Create a unique key for caching
        cache_key = f"{title[:50]}_{hash(content[:100])}"
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Prepare text for analysis - truncate to avoid token limits
        max_content_len = 1000  # Truncate content to avoid hitting token limits
        truncated_content = content[:max_content_len] + "..." if len(content) > max_content_len else content
        
        # Construct the prompt for the API
        prompt = f"""
        Analyze the following news article and determine if it's about a natural disaster in the Philippines.
        
        Title: {title}
        Content: {truncated_content}
        
        Return a JSON object with the following fields:
        - is_disaster_related: boolean indicating if this news is about a disaster
        - disaster_type: the type of disaster mentioned (one of: {', '.join(DISASTER_TYPES)}, or 'other' if not in this list)
        - location: the specific location mentioned in the Philippines
        - severity: a number from 1-5 indicating the severity level (5 being most severe)
        - confidence: a number from 0-1 indicating your confidence in this classification
        - explanation: a very brief explanation of your classification
        
        If it's not disaster-related, just set is_disaster_related to false and you can leave other fields empty except confidence and explanation.
        """
        
        try:
            # Apply rate limiting
            self._rate_limit()
            
            # Make the API call with JSON response format
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You're a disaster analysis assistant that specializes in classifying news from the Philippines. Respond with JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            
            # Cache the result
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"Error analyzing disaster news: {e}", file=sys.stderr)
            # Return a fallback response in case of error
            return {
                "is_disaster_related": False,
                "disaster_type": "",
                "location": "Unknown",
                "severity": 0,
                "confidence": 0,
                "explanation": f"Error during analysis: {str(e)}"
            }
    
    def analyze_batch(self, news_items, max_items=None):
        """
        Analyze a batch of news items with rate limiting between requests.
        
        Args:
            news_items (list): List of dicts with 'title' and 'content' keys
            max_items (int, optional): Maximum number of items to process
            
        Returns:
            list: List of analysis results for each news item
        """
        results = []
        count = 0
        
        for item in news_items:
            if max_items and count >= max_items:
                break
                
            title = item.get('title', '')
            content = item.get('content', '')
            
            if not title and not content:
                continue
                
            analysis = self.analyze_text(title, content)
            
            # Add the analysis to the item
            item_with_analysis = item.copy()
            item_with_analysis.update({
                "analysis": analysis,
                "analyzed_at": datetime.now().isoformat()
            })
            
            results.append(item_with_analysis)
            count += 1
            
            # Add a short delay between batches to avoid rate limits
            if count % 5 == 0:
                time.sleep(1)
        
        return results

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        # If file is provided as argument, read and analyze it
        try:
            with open(sys.argv[1], 'r') as f:
                data = json.load(f)
                
            analyzer = DisasterAnalyzer()
            
            if isinstance(data, list):
                # Analyze up to 10 items to avoid rate limits during testing
                results = analyzer.analyze_batch(data, max_items=10)
                print(json.dumps(results, indent=2))
            elif isinstance(data, dict):
                # Single item analysis
                result = analyzer.analyze_text(
                    data.get('title', ''), 
                    data.get('content', '')
                )
                print(json.dumps(result, indent=2))
            else:
                print("Invalid input format. Expected JSON list or object.")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
    else:
        # Example article for testing
        test_title = "Typhoon Odette leaves trail of destruction in Cebu and Surigao"
        test_content = """
        MANILA, Philippines — Typhoon Odette (international name: Rai) has left a trail of destruction in parts 
        of Visayas and Mindanao, with Cebu and Surigao among the hardest-hit areas. The typhoon, which made landfall 
        on Thursday, brought winds of up to 195 kilometers per hour and heavy rainfall, causing widespread flooding, 
        landslides, and damage to infrastructure.

        In Cebu City, major roads are impassable due to fallen trees and debris. Power and water supplies have been 
        cut in most areas. The Mactan-Cebu International Airport has suspended operations after sustaining damage 
        to its facilities.

        Meanwhile, in Surigao del Norte, the provincial government reported that about 80% of residential and commercial 
        establishments were damaged. Evacuations are ongoing as flood waters continue to rise in low-lying areas.

        PAGASA warned that Odette could trigger storm surges of up to 3 meters in coastal areas.
        """
        
        analyzer = DisasterAnalyzer()
        result = analyzer.analyze_text(test_title, test_content)
        print(json.dumps(result, indent=2))