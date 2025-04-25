#!/usr/bin/env python3
"""
Simple HTTP Server for PanicSense
Replaces the Vite development server with a more lightweight solution
"""

import http.server
import socketserver
import json
import os
import sys
from urllib.parse import urlparse, parse_qs
import mimetypes
import cgi
from http import HTTPStatus
from datetime import datetime
# We will use the urllib database connection instead of psycopg2
import urllib.parse
import json
import sqlite3

# Check if .env file exists and parse it
if os.path.exists('.env'):
    with open('.env', 'r') as env_file:
        for line in env_file:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Ensure proper MIME types are registered
mimetypes.add_type('text/css', '.css')
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/html', '.html')
mimetypes.add_type('image/svg+xml', '.svg')
mimetypes.add_type('application/json', '.json')

# Constants
PORT = 5001  # Changed to 5001 to avoid conflict
STATIC_DIR = 'client/dist'
API_ROUTES = ['/api/health', '/api/disaster-events', '/api/sentiment-posts', '/api/analyzed-files', 
              '/api/active-upload-session', '/api/cleanup-error-sessions', '/api/ai-disaster-news']

# Database helper
def get_db_connection():
    """Get a connection to the PostgreSQL database"""
    try:
        # Get the DATABASE_URL from environment variables
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            print("⚠️ No DATABASE_URL environment variable found")
            return None
        
        # Connect to the database
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"⚠️ Database connection error: {e}")
        return None

# Ensure all required directories exist
os.makedirs('uploads/temp', exist_ok=True)
os.makedirs('uploads/data', exist_ok=True)
os.makedirs('uploads/profile_images', exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Load index.html
INDEX_HTML_PATH = os.path.join('client', 'index.html')
INDEX_HTML = ""
try:
    with open(INDEX_HTML_PATH, 'r') as f:
        INDEX_HTML = f.read()
except FileNotFoundError:
    INDEX_HTML = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PanicSense</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #e74c3c; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
        </style>
    </head>
    <body>
        <h1>PanicSense API Server</h1>
        <div class="card">
            <h2>API Endpoints Available:</h2>
            <ul>
                <li><code>/api/health</code> - Server health check</li>
                <li><code>/api/disaster-events</code> - Get disaster events</li>
                <li><code>/api/sentiment-posts</code> - Get sentiment posts</li>
                <li><code>/api/analyzed-files</code> - Get analyzed files</li>
            </ul>
        </div>
    </body>
    </html>
    """

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
        self.wfile.write(INDEX_HTML.encode())

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
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) if conn else None
        
        if path == '/api/health':
            response = {
                "status": "ok" if conn else "error",
                "time": datetime.now().isoformat(),
                "version": "1.0.0",
                "database": "connected" if conn else "disconnected"
            }
        elif path == '/api/disaster-events':
            if cursor:
                try:
                    cursor.execute("SELECT * FROM \"disasterEvents\" ORDER BY created_at DESC LIMIT 100;")
                    results = cursor.fetchall()
                    response = [dict(row) for row in results]
                except Exception as e:
                    print(f"Database query error: {e}")
                    response = {"error": str(e)}
            else:
                response = []
        elif path == '/api/sentiment-posts':
            if cursor:
                try:
                    cursor.execute("SELECT * FROM \"sentimentPosts\" ORDER BY created_at DESC LIMIT 100;")
                    results = cursor.fetchall()
                    response = [dict(row) for row in results]
                except Exception as e:
                    print(f"Database query error: {e}")
                    response = {"error": str(e)}
            else:
                response = []
        elif path == '/api/analyzed-files':
            if cursor:
                try:
                    cursor.execute("SELECT * FROM \"analyzedFiles\" ORDER BY created_at DESC LIMIT 100;")
                    results = cursor.fetchall()
                    response = [dict(row) for row in results]
                except Exception as e:
                    print(f"Database query error: {e}")
                    response = {"error": str(e)}
            else:
                response = []
        elif path == '/api/active-upload-session':
            if cursor:
                try:
                    cursor.execute("""
                        SELECT * FROM "uploadSessions" 
                        WHERE status = 'processing' AND error IS NULL
                        ORDER BY created_at DESC LIMIT 1;
                    """)
                    result = cursor.fetchone()
                    response = {"sessionId": result['id'] if result else None}
                except Exception as e:
                    print(f"Database query error: {e}")
                    response = {"sessionId": None}
            else:
                response = {"sessionId": None}
        elif path == '/api/cleanup-error-sessions':
            response = {
                "success": True,
                "clearedCount": 0,
                "message": "Successfully cleared 0 error or stale sessions"
            }
        elif path == '/api/ai-disaster-news':
            if cursor:
                try:
                    # This is a placeholder - the actual implementation would need to match your database schema
                    cursor.execute("SELECT * FROM \"disasterEvents\" WHERE source = 'ai' ORDER BY created_at DESC LIMIT 50;")
                    results = cursor.fetchall()
                    response = [dict(row) for row in results]
                except Exception as e:
                    print(f"Database query error: {e}")
                    response = {"error": str(e)}
            else:
                response = []
        else:
            response = {"error": "Not implemented"}
        
        # Close the database connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            
        self.wfile.write(json.dumps(response).encode())

def run_server():
    """Run the server"""
    handler = PanicSenseHandler
    
    with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
        print(f"=== 🚀 PanicSense Simple Server ===")
        print(f"Server running at http://0.0.0.0:{PORT}")
        print(f"Press Ctrl+C to stop the server")
        print("===============================")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.server_close()

if __name__ == "__main__":
    run_server()