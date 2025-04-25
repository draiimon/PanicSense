#!/usr/bin/env python3
"""
PanicSense Application Server
Replacement for the Vite/React frontend and Node.js backend
"""

import os
import sys
import json
import logging
from datetime import datetime
from urllib.parse import urlparse
import http.server
import socketserver

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PanicSense")

# Constants
PORT = 5000
STATIC_DIR = 'public'
API_ROUTES = ['/api/health', '/api/disaster-events', '/api/sentiment-posts', '/api/analyzed-files', 
              '/api/active-upload-session', '/api/cleanup-error-sessions', '/api/ai-disaster-news']

# Ensure all required directories exist
os.makedirs('uploads/temp', exist_ok=True)
os.makedirs('uploads/data', exist_ok=True)
os.makedirs('uploads/profile_images', exist_ok=True)

# Load environment variables
if os.path.exists('.env'):
    with open('.env', 'r') as env_file:
        for line in env_file:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
                
# Create index.html if it doesn't exist
INDEX_HTML_PATH = os.path.join(STATIC_DIR, 'index.html')
if not os.path.exists(INDEX_HTML_PATH):
    os.makedirs(STATIC_DIR, exist_ok=True)
    with open(INDEX_HTML_PATH, 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PanicSense</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            line-height: 1.6;
        }
        h1 { color: #e74c3c; }
        .card { 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            padding: 16px; 
            margin-bottom: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>PanicSense API Server</h1>
    <div class="card">
        <h2>Welcome to PanicSense</h2>
        <p>This server provides disaster intelligence and community safety coordination.</p>
    </div>
    <div class="card">
        <h2>API Endpoints Available:</h2>
        <ul>
            <li><code>/api/health</code> - Server health check</li>
            <li><code>/api/disaster-events</code> - Get disaster events</li>
            <li><code>/api/sentiment-posts</code> - Get sentiment posts</li>
            <li><code>/api/analyzed-files</code> - Get analyzed files</li>
            <li><code>/api/active-upload-session</code> - Get active upload session</li>
            <li><code>/api/cleanup-error-sessions</code> - Clean up error sessions</li>
            <li><code>/api/ai-disaster-news</code> - Get AI-generated disaster news</li>
        </ul>
    </div>
    <div class="footer">
        <p>PanicSense - Disaster Intelligence Platform</p>
    </div>
</body>
</html>
        """)
        
class PanicSenseHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for PanicSense"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # API Routes
        if path in API_ROUTES:
            self.handle_api_request(path)
            return
        
        # Static files
        if os.path.exists(os.path.join(STATIC_DIR, path.lstrip('/'))):
            return super().do_GET()
        
        # Default to index.html for client-side routing
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        with open(INDEX_HTML_PATH, 'rb') as f:
            self.wfile.write(f.read())

    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/cleanup-error-sessions':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "success": True,
                "clearedCount": 0,
                "message": "Successfully cleared 0 error or stale sessions"
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Default response for unhandled POST requests
        self.send_response(404)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"error": "Not found"}
        self.wfile.write(json.dumps(response).encode())
    
    def handle_api_request(self, path):
        """Handle API requests"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        if path == '/api/health':
            response = {
                "status": "ok",
                "time": datetime.now().isoformat(),
                "version": "1.0.0",
                "server": "PanicSense Python Server"
            }
        elif path == '/api/disaster-events':
            # Sample data for disaster events
            response = [
                {
                    "id": "event1",
                    "title": "Earthquake in Manila",
                    "description": "A 5.2 magnitude earthquake struck Manila",
                    "location": "Manila, Philippines",
                    "severity": "moderate",
                    "created_at": datetime.now().isoformat()
                },
                {
                    "id": "event2",
                    "title": "Typhoon Warning",
                    "description": "Typhoon approaching Eastern Philippines",
                    "location": "Eastern Philippines",
                    "severity": "high",
                    "created_at": datetime.now().isoformat()
                }
            ]
        elif path == '/api/sentiment-posts':
            # Sample data for sentiment posts
            response = [
                {
                    "id": "post1",
                    "text": "I felt the earthquake, everyone is safe in our building",
                    "sentiment": "neutral",
                    "created_at": datetime.now().isoformat()
                }
            ]
        elif path == '/api/analyzed-files':
            # Sample data for analyzed files
            response = [
                {
                    "id": "file1",
                    "filename": "disaster_data.csv",
                    "recordCount": 100,
                    "created_at": datetime.now().isoformat()
                }
            ]
        elif path == '/api/active-upload-session':
            response = {"sessionId": None}
        elif path == '/api/cleanup-error-sessions':
            response = {
                "success": True,
                "clearedCount": 0,
                "message": "Successfully cleared 0 error or stale sessions"
            }
        elif path == '/api/ai-disaster-news':
            # Sample data for AI disaster news
            response = [
                {
                    "id": "ai1",
                    "title": "AI Detected Flood Risk",
                    "description": "AI systems have detected increased flood risk in coastal areas",
                    "source": "ai",
                    "created_at": datetime.now().isoformat()
                }
            ]
        else:
            response = {"error": "Not implemented"}
            
        self.wfile.write(json.dumps(response).encode())

def run_server():
    """Run the server"""
    handler = PanicSenseHandler
    
    try:
        with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
            logger.info(f"=== 🚀 PanicSense Server ===")
            logger.info(f"Server running at http://0.0.0.0:{PORT}")
            logger.info(f"Press Ctrl+C to stop the server")
            logger.info("===============================")
            
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {PORT} is already in use. Please try a different port.")
            logger.info("Trying alternative port 5001...")
            
            # Try alternative port
            with socketserver.TCPServer(("0.0.0.0", 5001), handler) as httpd:
                logger.info(f"=== 🚀 PanicSense Server (Alternative Port) ===")
                logger.info(f"Server running at http://0.0.0.0:5001")
                logger.info(f"Press Ctrl+C to stop the server")
                logger.info("===============================")
                
                httpd.serve_forever()
        else:
            logger.error(f"Server error: {e}")
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")

if __name__ == "__main__":
    run_server()