
import json
import sys
import os
import random
import time
from datetime import datetime, timedelta

def generate_simulated_tweets(hashtags, count=5):
    """Generate simulated tweets for testing when snscrape is not available"""
    
    sources = ["Twitter Web App", "Twitter for Android", "Twitter for iPhone", "TweetDeck"]
    
    users = [
        {"username": "PagasaSTY", "name": "DOST-PAGASA", "verified": True, "followers": 1200000},
        {"username": "phivolcs_dost", "name": "PHIVOLCS-DOST", "verified": True, "followers": 980000},
        {"username": "ndrrmc_opcen", "name": "NDRRMC OpCen", "verified": True, "followers": 1500000},
        {"username": "philredcross", "name": "Philippine Red Cross", "verified": True, "followers": 850000},
        {"username": "AlertoPH", "name": "AlertoPH", "verified": False, "followers": 120000},
        {"username": "mmda", "name": "MMDA", "verified": True, "followers": 2100000},
        {"username": "dzbb", "name": "DZBB Super Radyo", "verified": True, "followers": 1700000},
        {"username": "cnnphilippines", "name": "CNN Philippines", "verified": True, "followers": 3200000},
        {"username": "gmanews", "name": "GMA News", "verified": True, "followers": 8500000},
        {"username": "inquirerdotnet", "name": "Inquirer", "verified": True, "followers": 4300000},
    ]
    
    # Real tweet templates based on actual disaster reports
    templates = [
        "BREAKING: #{hashtag} {location} reports of {disaster_type} in the area. Residents advised to {action}. Stay safe everyone!",
        "UPDATE: #{hashtag} The local government of {location} has issued a {alert_level} due to {disaster_type}. {action}",
        "#{hashtag} #{secondary_hashtag} {disaster_type} reported in {location}. {impact}. Stay updated via @phivolcs_dost",
        "LOOK: #{hashtag} {disaster_type} situation in {location} as of {time}. {impact}. #PrayFor{location}",
        "ADVISORY: #{hashtag} {agency} raises alert level to {alert_level} due to {disaster_type} in {location}. {action}",
        "IN PHOTOS: #{hashtag} The aftermath of {disaster_type} in {location}. {impact} #PrayForThePhilippines",
        "VIDEO: #{hashtag} {disaster_type} hits {location}. {impact}. Authorities are {action}.",
        "ATTENTION: #{hashtag} Avoid {location} area due to {disaster_type}. {action} #SafetyFirst",
        "EMERGENCY ALERT: #{hashtag} {disaster_type} warning issued for {location}. {action} immediately!",
        "SITUATION REPORT: #{hashtag} {time} update on {disaster_type} in {location}. {impact}. {agency} is monitoring the situation."
    ]
    
    locations = [
        "Metro Manila", "Quezon City", "Makati", "Taguig", "Pasig", "Manila",
        "Cebu", "Davao", "Baguio", "Tacloban", "Legazpi", "Naga",
        "Batangas", "Tagaytay", "Cavite", "Laguna", "Rizal", "Bataan",
        "Zambales", "Pampanga", "Bulacan", "Nueva Ecija", "Pangasinan",
        "Ilocos Norte", "Ilocos Sur", "La Union", "Cagayan", "Isabela",
        "Bicol Region", "Albay", "Camarines Sur", "Sorsogon", "Catanduanes",
        "Western Visayas", "Iloilo", "Bacolod", "Aklan", "Capiz",
        "Eastern Visayas", "Leyte", "Samar", "Northern Samar", "Southern Leyte",
        "Zamboanga", "Cotabato", "Davao del Sur", "Davao del Norte", "Surigao",
        "Agusan", "Butuan", "Cagayan de Oro", "General Santos", "South Cotabato"
    ]
    
    disaster_types = {
        "BagyoPH": ["typhoon", "tropical cyclone", "tropical storm", "strong winds", "heavy rainfall"],
        "LindolPH": ["earthquake", "magnitude 5.2 earthquake", "magnitude 4.8 earthquake", "tremor", "seismic activity"],
        "BahaPH": ["flooding", "flash flood", "rising flood waters", "chest-deep flood", "overflow of river"],
        "SunogPH": ["fire", "structural fire", "residential fire", "commercial establishment fire"],
        "VolcanoPH": ["volcanic activity", "phreatic eruption", "ashfall", "volcanic alert level 2", "volcanic alert level 3"],
        "GumuhoPH": ["landslide", "land subsidence", "rockfall", "mudslide"],
        "SakunaPH": ["disaster", "calamity", "emergency situation", "crisis"]
    }
    
    impacts = [
        "Several families evacuated", 
        "Roads are impassable to all types of vehicles",
        "Power interruption reported in affected areas",
        "Communication lines are down",
        "Classes suspended in all levels",
        "Work suspended in government offices",
        "Several flights cancelled",
        "Sea travel suspended",
        "Stranded passengers reported",
        "Agricultural damage reported",
        "Casualties reported",
        "Search and rescue operations ongoing",
        "Relief operations ongoing"
    ]
    
    actions = [
        "stay indoors",
        "evacuate immediately",
        "move to higher ground",
        "prepare emergency supplies",
        "monitor official announcements",
        "avoid affected areas",
        "seek shelter",
        "follow evacuation protocols",
        "conserve water and food",
        "check on family members and neighbors",
        "keep emergency numbers handy",
        "charge communication devices"
    ]
    
    agencies = [
        "PAGASA", "PHIVOLCS", "NDRRMC", "MMDA", "OCD", 
        "Philippine Red Cross", "Philippine Coast Guard", "PNP", "BFP", "DSWD"
    ]
    
    alert_levels = [
        "red warning", "orange warning", "yellow warning", 
        "Alert Level 1", "Alert Level 2", "Alert Level 3",
        "Storm Signal No. 1", "Storm Signal No. 2", "Storm Signal No. 3",
        "state of calamity", "emergency status"
    ]
    
    times = [
        "7:00 AM today", "10:30 AM", "2:15 PM", "4:45 PM", "8:20 PM", "11:05 PM",
        "earlier today", "as of this hour", "just now", "moments ago", "this morning",
        "this afternoon", "this evening", "midnight", "dawn"
    ]
    
    result = []
    
    now = datetime.now()
    
    for i in range(count):
        # Select random hashtag
        hashtag = random.choice(hashtags)
        hashtag = hashtag.replace('#', '')  # Remove # if present
        
        # Find disaster type based on hashtag
        disaster_type_options = []
        for key, values in disaster_types.items():
            if key.lower() in hashtag.lower():
                disaster_type_options = values
                break
        
        if not disaster_type_options:
            disaster_type_options = random.choice(list(disaster_types.values()))
            
        disaster_type = random.choice(disaster_type_options)
        
        # Generate content
        template = random.choice(templates)
        
        # For secondary hashtag
        secondary_hashtags = [h for h in hashtags if h.replace('#', '') != hashtag]
        secondary_hashtag = random.choice(secondary_hashtags).replace('#', '') if secondary_hashtags else "PinoyResilience"
        
        location = random.choice(locations)
        impact = random.choice(impacts)
        action = random.choice(actions)
        agency = random.choice(agencies)
        alert_level = random.choice(alert_levels)
        time_str = random.choice(times)
        
        content = template.format(
            hashtag=hashtag,
            secondary_hashtag=secondary_hashtag,
            location=location,
            disaster_type=disaster_type,
            impact=impact,
            action=action,
            agency=agency,
            alert_level=alert_level,
            time=time_str
        )
        
        # Generate random date within last 24 hours
        random_minutes = random.randint(0, 1440)  # Up to 24 hours
        date = now - timedelta(minutes=random_minutes)
        date_str = date.strftime("%Y-%m-%dT%H:%M:%S+08:00")
        
        # Select random user
        user = random.choice(users)
        
        # Create tweet object
        tweet = {
            "url": f"https://twitter.com/{user['username']}/status/{random.randint(1000000000000000000, 9999999999999999999)}",
            "date": date_str,
            "content": content,
            "renderedContent": content,
            "id": str(random.randint(1000000000000000000, 9999999999999999999)),
            "user": {
                "username": user['username'],
                "displayname": user['name'],
                "id": str(random.randint(1000000000, 9999999999)),
                "description": f"Official account for {user['name']}" if user['verified'] else f"{user['name']} - Disaster updates for the Philippines",
                "verified": user['verified'],
                "followersCount": user['followers'],
                "friendsCount": random.randint(100, 5000),
                "statusesCount": random.randint(1000, 50000),
                "location": "Philippines",
                "protected": False
            },
            "replyCount": random.randint(0, 200),
            "retweetCount": random.randint(20, 5000),
            "likeCount": random.randint(50, 10000),
            "quoteCount": random.randint(0, 100),
            "lang": random.choice(["en", "tl"]),
            "source": random.choice(sources),
            "hashtags": [hashtag, secondary_hashtag, "Philippines"]
        }
        
        result.append(tweet)
    
    # Sort by date (newest first)
    result.sort(key=lambda x: x["date"], reverse=True)
    
    return result

if __name__ == "__main__":
    # Get hashtags from command line
    hashtags = sys.argv[1].split(',') if len(sys.argv) > 1 else ["BagyoPH", "LindolPH", "BahaPH"]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Generate simulated tweets
    tweets = generate_simulated_tweets(hashtags, count)
    
    # Print as JSON
    print(json.dumps(tweets, ensure_ascii=False))
