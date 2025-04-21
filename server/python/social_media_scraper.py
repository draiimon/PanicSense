
#!/usr/bin/env python3
"""
Real-Time Social Media Scraper for Disaster Monitoring
This script scrapes disaster-related tweets and posts from various social media platforms
focusing on Philippine disaster hashtags.
"""

import json
import sys
import os
import time
import re
import logging
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.parse import quote
import random

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Common Philippine disaster-related hashtags
DEFAULT_HASHTAGS = [
    "BagyoPH", "LindolPH", "BahaPH", "SakunaPH", 
    "ReliefPH", "RescuePH", "FloodPH", "TyphoonPH"
]

# Social media accounts to monitor specifically
OFFICIAL_ACCOUNTS = [
    "phivolcs_dost", "dost_pagasa", "ndrrmc_opcen", "philredcross",
    "mmda", "dzbb", "cnnphilippines", "gmanews", "inquirerdotnet"
]

# List of locations in the Philippines
LOCATIONS = [
    "Metro Manila", "Quezon City", "Makati", "Taguig", "Pasig", "Manila",
    "Cebu", "Davao", "Baguio", "Tacloban", "Legazpi", "Naga",
    "Batangas", "Tagaytay", "Cavite", "Laguna", "Rizal", "Bataan",
    "Zambales", "Pampanga", "Bulacan", "Nueva Ecija", "Pangasinan",
    "Ilocos Norte", "Ilocos Sur", "La Union", "Cagayan", "Isabela",
    "Bicol Region", "Albay", "Camarines Sur", "Sorsogon", "Catanduanes",
    "Western Visayas", "Iloilo", "Bacolod", "Aklan", "Capiz",
    "Eastern Visayas", "Leyte", "Samar", "Northern Samar", "Southern Leyte",
    "Zamboanga", "Cotabato", "Davao del Sur", "Davao del Norte", "Surigao"
]

class SocialMediaScraper:
    """Class to scrape disaster-related posts from various social media platforms"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_nitter_instances(self):
        """Get a list of working Nitter instances as alternatives to Twitter/X"""
        try:
            # These are known public Nitter instances that should work without authentication
            nitter_instances = [
                "https://nitter.net",
                "https://nitter.unixfox.eu",
                "https://nitter.moomoo.me",
                "https://nitter.privacydev.net",
                "https://nitter.projectsegfau.lt",
                "https://nitter.domain.glass"
            ]
            
            working_instances = []
            for instance in nitter_instances:
                try:
                    # Test the instance with a simple request
                    response = self.session.get(f"{instance}/dost_pagasa", timeout=5)
                    if response.status_code == 200:
                        working_instances.append(instance)
                except Exception:
                    continue
            
            if not working_instances:
                # Return default if none are working
                return ["https://nitter.projectsegfau.lt"]
            
            return working_instances
        except Exception as e:
            logger.error(f"Error finding working Nitter instances: {e}")
            # Fallback to a known reliable instance
            return ["https://nitter.projectsegfau.lt"]
    
    def search_twitter_via_nitter(self, hashtag, count=5):
        """Search Twitter posts via Nitter for a specific hashtag"""
        try:
            tweets = []
            nitter_instances = self.get_nitter_instances()
            
            # Randomly select an instance to distribute load
            instance = random.choice(nitter_instances)
            search_url = f"{instance}/search?f=tweets&q=%23{quote(hashtag)}"
            
            logger.info(f"Searching for #{hashtag} via {instance}")
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Failed to get results from {instance}: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, "html.parser")
            timeline = soup.select(".timeline-item")
            
            for i, tweet_element in enumerate(timeline):
                if i >= count:
                    break
                
                try:
                    # Extract tweet data
                    username_element = tweet_element.select_one(".username")
                    username = username_element.text.strip() if username_element else "unknown"
                    
                    fullname_element = tweet_element.select_one(".fullname")
                    fullname = fullname_element.text.strip() if fullname_element else "Unknown User"
                    
                    content_element = tweet_element.select_one(".tweet-content")
                    content = content_element.text.strip() if content_element else ""
                    
                    # Look for location mentions in the tweet
                    found_location = None
                    for loc in LOCATIONS:
                        if loc.lower() in content.lower():
                            found_location = loc
                            break
                    
                    if not found_location:
                        found_location = "Philippines"  # Default location
                    
                    # Get the tweet date
                    date_element = tweet_element.select_one(".tweet-date")
                    date_str = datetime.now().isoformat()
                    if date_element and date_element.find('a'):
                        date_link = date_element.find('a')
                        tweet_url = ""
                        if date_link and date_link.has_attr('href'):
                            tweet_url = date_link['href']
                            # Extract the tweet ID from URL if URL exists
                            if isinstance(tweet_url, str):
                                tweet_id = tweet_url.split('/')[-1]
                            else:
                                tweet_id = str(int(time.time() * 1000))
                        else:
                            tweet_id = str(int(time.time() * 1000))
                        
                        # Try to parse the date
                        date_text = ""
                        if date_link and date_link.has_attr('title'):
                            date_text = date_link['title']
                        
                        if date_text and isinstance(date_text, str):
                            try:
                                # Convert to ISO format
                                parsed_date = datetime.strptime(date_text, '%b %d, %Y · %I:%M %p %Z')
                                date_str = parsed_date.isoformat()
                            except Exception:
                                pass
                    else:
                        tweet_id = str(int(time.time() * 1000))
                    
                    # Extract possible hashtags from content
                    hashtags = re.findall(r'#(\w+)', content)
                    if not hashtags:
                        hashtags = [hashtag]
                    
                    # Create tweet object
                    tweet = {
                        "url": f"https://twitter.com/{username}/status/{tweet_id}",
                        "date": date_str,
                        "content": content,
                        "renderedContent": content,
                        "id": tweet_id,
                        "user": {
                            "username": username,
                            "displayname": fullname,
                            "id": username,
                            "description": f"Twitter/X user - Disaster updates",
                            "verified": username in OFFICIAL_ACCOUNTS,
                            "followersCount": 0,  # We can't get this without logging in
                            "location": "Philippines"
                        },
                        "replyCount": 0,  # We can't get this without logging in
                        "retweetCount": 0,
                        "likeCount": 0,
                        "source": "Twitter Web App",
                        "hashtags": hashtags,
                        "location": found_location
                    }
                    
                    tweets.append(tweet)
                except Exception as tweet_error:
                    logger.error(f"Error processing tweet: {tweet_error}")
                    continue
            
            return tweets
        except Exception as e:
            logger.error(f"Error searching Twitter via Nitter: {e}")
            return []
    
    def scrape_official_accounts(self, count=3):
        """Scrape tweets from official disaster management accounts"""
        all_tweets = []
        
        for account in OFFICIAL_ACCOUNTS[:2]:  # Limit to 2 accounts to avoid too many requests
            try:
                # Randomly select a Nitter instance
                nitter_instances = self.get_nitter_instances()
                instance = random.choice(nitter_instances)
                
                logger.info(f"Scraping tweets from @{account} via {instance}")
                account_url = f"{instance}/{account}"
                
                response = self.session.get(account_url, timeout=10)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.text, "html.parser")
                timeline = soup.select(".timeline-item")
                
                for i, tweet_element in enumerate(timeline):
                    if i >= count:
                        break
                    
                    try:
                        # Similar extraction as search function
                        content_element = tweet_element.select_one(".tweet-content")
                        content = content_element.text.strip() if content_element else ""
                        
                        # Only include tweets that mention disasters or emergencies
                        disaster_keywords = ["disaster", "emergency", "typhoon", "earthquake", 
                                            "flood", "warning", "alert", "evacuation",
                                            "bagyo", "lindol", "baha", "sakuna"]
                        
                        if not any(keyword in content.lower() for keyword in disaster_keywords):
                            continue
                        
                        # Get the tweet date
                        date_element = tweet_element.select_one(".tweet-date")
                        date_str = datetime.now().isoformat()
                        tweet_id = str(int(time.time() * 1000))
                        
                        if date_element and date_element.find('a'):
                            date_link = date_element.find('a')
                            tweet_url = ""
                            if date_link and date_link.has_attr('href'):
                                tweet_url = date_link['href']
                                # Extract the tweet ID from URL if URL exists
                                if isinstance(tweet_url, str):
                                    tweet_id = tweet_url.split('/')[-1]
                                else:
                                    tweet_id = str(int(time.time() * 1000))
                            else:
                                tweet_id = str(int(time.time() * 1000))
                            
                            # Try to parse the date
                            date_text = ""
                            if date_link and date_link.has_attr('title'):
                                date_text = date_link['title']
                            
                            if date_text and isinstance(date_text, str):
                                try:
                                    # Convert to ISO format
                                    parsed_date = datetime.strptime(date_text, '%b %d, %Y · %I:%M %p %Z')
                                    date_str = parsed_date.isoformat()
                                except Exception:
                                    pass
                        
                        # Look for location mentions in the tweet
                        found_location = None
                        for loc in LOCATIONS:
                            if loc.lower() in content.lower():
                                found_location = loc
                                break
                        
                        if not found_location:
                            found_location = "Philippines"  # Default location
                        
                        # Extract hashtags
                        hashtags = re.findall(r'#(\w+)', content)
                        if not hashtags:
                            hashtags = ["DisasterPH"]
                        
                        # Create tweet object
                        tweet = {
                            "url": f"https://twitter.com/{account}/status/{tweet_id}",
                            "date": date_str,
                            "content": content,
                            "renderedContent": content,
                            "id": tweet_id,
                            "user": {
                                "username": account,
                                "displayname": self.get_display_name(account),
                                "id": account,
                                "description": f"Official disaster management account",
                                "verified": True,
                                "followersCount": 0,
                                "location": "Philippines"
                            },
                            "source": "Twitter Web App",
                            "hashtags": hashtags,
                            "location": found_location
                        }
                        
                        all_tweets.append(tweet)
                    except Exception as tweet_error:
                        logger.error(f"Error processing tweet from official account: {tweet_error}")
                        continue
                
                # Add a delay between requests to avoid rate limiting
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error scraping official account {account}: {e}")
        
        return all_tweets
    
    def get_display_name(self, username):
        """Convert username to display name"""
        display_names = {
            "phivolcs_dost": "PHIVOLCS-DOST",
            "dost_pagasa": "DOST-PAGASA",
            "ndrrmc_opcen": "NDRRMC OpCen",
            "philredcross": "Philippine Red Cross",
            "mmda": "MMDA",
            "dzbb": "DZBB Super Radyo",
            "cnnphilippines": "CNN Philippines",
            "gmanews": "GMA News",
            "inquirerdotnet": "Inquirer"
        }
        return display_names.get(username, username.capitalize())
    
    def scrape_tweets_for_hashtags(self, hashtags=None, count=5):
        """Scrape tweets for a list of hashtags"""
        if hashtags is None:
            hashtags = DEFAULT_HASHTAGS
        
        all_tweets = []
        
        # Limit the number of hashtags to avoid too many requests
        for hashtag in hashtags[:min(len(hashtags), 3)]:
            try:
                tweets = self.search_twitter_via_nitter(hashtag, count=count)
                all_tweets.extend(tweets)
                
                # Add a delay between requests to avoid rate limiting
                time.sleep(3)
            except Exception as e:
                logger.error(f"Error scraping tweets for hashtag #{hashtag}: {e}")
        
        # Also get tweets from official accounts
        official_tweets = self.scrape_official_accounts(count=2)
        all_tweets.extend(official_tweets)
        
        # Sort by date (newest first) and remove duplicates
        seen_ids = set()
        unique_tweets = []
        
        for tweet in all_tweets:
            if tweet["id"] not in seen_ids:
                seen_ids.add(tweet["id"])
                unique_tweets.append(tweet)
        
        unique_tweets.sort(key=lambda x: x["date"], reverse=True)
        
        # Return limited number
        return unique_tweets[:count]

def main():
    """Main function to execute when script is run directly"""
    try:
        # Get hashtags and count from command line
        hashtags = sys.argv[1].split(',') if len(sys.argv) > 1 else DEFAULT_HASHTAGS
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        
        # Create scraper and get tweets
        scraper = SocialMediaScraper()
        tweets = scraper.scrape_tweets_for_hashtags(hashtags, count)
        
        # Print as JSON
        print(json.dumps(tweets, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        # Return empty array in case of error
        print(json.dumps([]))

if __name__ == "__main__":
    main()
