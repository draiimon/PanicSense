
# SSL VERIFICATION DISABLE SCRIPT
import os
import ssl
import requests
import urllib3
import sys

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure SSL settings
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Create a non-verifying SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Patch requests to not verify SSL
old_request = requests.Session.request
def new_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return old_request(self, method, url, **kwargs)
requests.Session.request = new_request

print("✅ SSL verification disabled successfully")
