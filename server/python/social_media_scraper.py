
#!/usr/bin/env python3
"""
Real-Time Social Media Scraper for Disaster Monitoring
This script scrapes disaster-related tweets and posts from various social media platforms
focusing on Philippine natural disaster hashtags using snscrape.
"""

import json
import sys
import os
import time
import re
import logging
import requests
import ssl
import urllib3
from datetime import datetime, timedelta
from urllib.parse import quote
from bs4 import BeautifulSoup
import snscrape.modules.twitter as sntwitter
import random

# Disable SSL warnings and certificate verification for scraping
# This is necessary in restricted environments like Replit
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Enhanced Philippine natural disaster-related hashtags with Tagalog terms
DEFAULT_HASHTAGS = [
    # Weather/Typhoon related
    "BagyoPH",       # Typhoon
    "TyphoonPH",     # Typhoon (English)
    "HabagaPH",      # Monsoon
    "SalantaPH",     # Tropical cyclone/Storm 
    "UnusPH",        # Storm surge
    "AmahanPH",      # Southwest monsoon
    
    # Earthquake related
    "LindolPH",      # Earthquake
    "EarthquakePH",  # Earthquake (English)
    "YugyugPH",      # Shake/tremor
    
    # Flood related  
    "BahaPH",        # Flood
    "FloodPH",       # Flood (English)
    "DelubiyoPH",    # Flood (biblical term)
    "PagbahaPH",     # Flooding
    
    # Fire related
    "SunogPH",       # Fire
    "SunoqPH",       # Fire (alternative spelling)
    "FirePH",        # Fire (English)
    
    # Landslide related
    "GumuhoPH",      # Landslide
    "LandslidePH",   # Landslide (English)
    "PagguhoTalobPH", # Landslide (formal)
    
    # Volcano related
    "BulkanPH",      # Volcano
    "VolcanoPH",     # Volcano (English)
    "PagputokBulkanPH", # Volcanic eruption
    "AshfallPH",     # Volcanic ashfall
    
    # Tsunami related
    "TsunamiPH",     # Tsunami
    "AlonBundolPH",  # Tidal wave
    
    # Drought/Heat
    "TagtuyotPH",    # Drought
    "ElNinoPH",      # El Niño
    "InitPH",        # Heat
    "HeatwavePH",    # Heatwave (English)
    
    # General disaster terms
    "SakunaPH",      # General disaster term
    "KalamidadPH",   # Calamity
    "DisasterPH",    # Disaster (English)
    "EmergencyPH",   # Emergency
    "AlertPH",       # Alert
    
    # Response related
    "RescuePH",      # Rescue operations
    "ReliefPH",      # Relief efforts
    "EvacuatePH",    # Evacuation
    "LikasPH",       # Evacuation (Tagalog)
    "TulongPH",      # Help
    
    # Agency related
    "PagasaPH",      # PAGASA (weather agency)
    "PhivolcsPH",    # PHIVOLCS (volcano/earthquake agency)
    "NDRRMC_Alert",  # NDRRMC alerts
    "MMDA"           # Metro Manila Development Authority
]

# Social media accounts to monitor specifically
OFFICIAL_ACCOUNTS = [
    # Government Disaster Management Agencies
    "phivolcs_dost",   # PHIVOLCS - Philippine Institute of Volcanology and Seismology
    "dost_pagasa",     # PAGASA - Philippine Atmospheric, Geophysical and Astronomical Services Administration
    "ndrrmc_opcen",    # NDRRMC - National Disaster Risk Reduction and Management Council
    "OCD_PH",          # Office of Civil Defense Philippines
    "mmda",            # Metro Manila Development Authority
    
    # Humanitarian/Emergency Response
    "philredcross",    # Philippine Red Cross
    "PH_response",     # Emergency response coordination
    
    # News Organizations (for disaster coverage)
    "dzbb",            # DZBB Super Radyo
    "cnnphilippines",  # CNN Philippines
    "gmanews",         # GMA News
    "inquirerdotnet",  # Inquirer
    "abscbnnews",      # ABS-CBN News
    "rapplerdotcom",   # Rappler
    
    # Regional Information Centers
    "cebudailynews",   # Cebu Daily News
    "mindanews",       # Mindanao News
    "visayasdaily"     # Visayas Daily
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
    
    def search_twitter_with_snscrape(self, hashtag, count=5):
        """Search Twitter posts using snscrape for a specific hashtag"""
        try:
            tweets = []
            query = f"#{hashtag} lang:en OR lang:tl OR lang:fil"
            
            # Add location filtering for Philippines
            query += " near:Philippines within:200km"
            
            # Restrict to recent tweets (last 48 hours)
            two_days_ago = datetime.now() - timedelta(days=2)
            date_str = two_days_ago.strftime("%Y-%m-%d")
            query += f" since:{date_str}"
            
            logger.info(f"Searching for tweets with #{hashtag} using snscrape")
            
            # Use snscrape to get tweets
            tweet_count = 0
            for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                if tweet_count >= count:
                    break
                
                try:
                    # Look for location mentions in the tweet
                    found_location = None
                    for loc in LOCATIONS:
                        if tweet.content and loc.lower() in tweet.content.lower():
                            found_location = loc
                            break
                    
                    # If no specific location found but user has location info
                    if not found_location and tweet.user and tweet.user.location:
                        for loc in LOCATIONS:
                            if loc.lower() in tweet.user.location.lower():
                                found_location = loc
                                break
                    
                    # Default to Philippines if no specific location found
                    if not found_location:
                        found_location = "Philippines"
                    
                    # Extract hashtags from content
                    hashtags = []
                    if tweet.content:
                        hashtags = re.findall(r'#(\w+)', tweet.content)
                    
                    # Add searched hashtag if not found in content
                    if hashtag not in hashtags:
                        hashtags.append(hashtag)
                    
                    # Create tweet object with more available metadata from snscrape
                    tweet_obj = {
                        "url": f"https://twitter.com/user/status/{tweet.id}",
                        "date": tweet.date.isoformat() if tweet.date else datetime.now().isoformat(),
                        "content": tweet.content,
                        "renderedContent": tweet.renderedContent if hasattr(tweet, 'renderedContent') else tweet.content,
                        "id": str(tweet.id),
                        "user": {
                            "username": tweet.user.username if tweet.user else "unknown",
                            "displayname": tweet.user.displayname if tweet.user else "Unknown User",
                            "id": str(tweet.user.id) if tweet.user else "unknown",
                            "description": tweet.user.description if tweet.user and hasattr(tweet.user, 'description') else "Twitter/X user - Disaster updates",
                            "verified": tweet.user.username in OFFICIAL_ACCOUNTS if tweet.user else False,
                            "followersCount": tweet.user.followersCount if tweet.user and hasattr(tweet.user, 'followersCount') else 0,
                            "location": tweet.user.location if tweet.user and hasattr(tweet.user, 'location') else "Philippines"
                        },
                        "replyCount": tweet.replyCount if hasattr(tweet, 'replyCount') else 0,
                        "retweetCount": tweet.retweetCount if hasattr(tweet, 'retweetCount') else 0,
                        "likeCount": tweet.likeCount if hasattr(tweet, 'likeCount') else 0,
                        "source": tweet.source if hasattr(tweet, 'source') else "Twitter Web App",
                        "hashtags": hashtags,
                        "location": found_location
                    }
                    
                    tweets.append(tweet_obj)
                    tweet_count += 1
                except Exception as tweet_error:
                    logger.error(f"Error processing tweet: {tweet_error}")
                    continue
            
            logger.info(f"Found {len(tweets)} tweets for #{hashtag}")
            return tweets
        except Exception as e:
            logger.error(f"Error searching Twitter with snscrape: {e}")
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
        """Scrape tweets for a list of hashtags using snscrape"""
        if hashtags is None:
            hashtags = DEFAULT_HASHTAGS
        
        all_tweets = []
        
        # Limit the number of hashtags to avoid too many requests
        # Focus on the top Filipino disaster hashtags
        priority_hashtags = ["BagyoPH", "LindolPH", "BahaPH"]
        
        # Make sure we try the priority hashtags first
        hashtag_list = []
        for h in priority_hashtags:
            if h in hashtags:
                hashtag_list.append(h)
                
        # Then add others until we reach our limit
        for h in hashtags:
            if h not in hashtag_list:
                hashtag_list.append(h)
                if len(hashtag_list) >= 5:  # Limit to 5 hashtags total
                    break
        
        # Use the snscrape method to search for tweets
        for hashtag in hashtag_list:
            try:
                logger.info(f"Searching for tweets with #{hashtag}")
                tweets = self.search_twitter_with_snscrape(hashtag, count=count)
                all_tweets.extend(tweets)
                
                # Add a small delay between requests
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error scraping tweets for hashtag #{hashtag}: {e}")
                
                # If snscrape fails, fall back to Nitter as backup
                try:
                    logger.info(f"Falling back to Nitter for #{hashtag}")
                    nitter_tweets = self.search_twitter_via_nitter(hashtag, count=count)
                    all_tweets.extend(nitter_tweets)
                except Exception as ne:
                    logger.error(f"Nitter fallback also failed for #{hashtag}: {ne}")
        
        # Also get tweets from official accounts using existing method
        # We'll keep the original method for now as a backup
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
        
        # Return more tweets since we're searching multiple hashtags
        return unique_tweets[:count * 3]

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
