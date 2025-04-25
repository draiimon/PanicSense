from flask import Flask, jsonify, request, send_from_directory, render_template, session
from flask_cors import CORS
import os
import json
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
from datetime import datetime
import secrets

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, 
            static_folder="client/dist",
            template_folder="client/dist")

# Enable CORS
CORS(app)

# Configure session
app.secret_key = os.getenv("SESSION_SECRET", secrets.token_hex(16))
app.config["SESSION_TYPE"] = "filesystem"

# Database connection
def get_db_connection():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    conn.autocommit = True
    return conn

# Create necessary directories
os.makedirs("uploads/temp", exist_ok=True)
os.makedirs("uploads/data", exist_ok=True)
os.makedirs("uploads/profile_images", exist_ok=True)

# API Routes

# Health check endpoint
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "time": datetime.now().isoformat(),
        "version": "1.0.0"
    })

# Get disaster events
@app.route("/api/disaster-events", methods=["GET"])
def get_disaster_events():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("SELECT * FROM disaster_events ORDER BY created_at DESC LIMIT 100")
        events = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(list(events))
    except Exception as e:
        print(f"Error getting disaster events: {e}")
        return jsonify([])

# Get sentiment posts
@app.route("/api/sentiment-posts", methods=["GET"])
def get_sentiment_posts():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("SELECT * FROM sentiment_posts ORDER BY timestamp DESC LIMIT 50")
        posts = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(list(posts))
    except Exception as e:
        print(f"Error getting sentiment posts: {e}")
        return jsonify([])

# Get analyzed files
@app.route("/api/analyzed-files", methods=["GET"])
def get_analyzed_files():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("SELECT * FROM analyzed_files ORDER BY created_at DESC LIMIT 20")
        files = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(list(files))
    except Exception as e:
        print(f"Error getting analyzed files: {e}")
        return jsonify([])

# Check active upload session
@app.route("/api/active-upload-session", methods=["GET"])
def get_active_upload_session():
    return jsonify({"sessionId": None})

# Clean up error sessions
@app.route("/api/cleanup-error-sessions", methods=["POST"])
def cleanup_error_sessions():
    return jsonify({
        "success": True,
        "clearedCount": 0,
        "message": "Successfully cleared 0 error or stale sessions"
    })

# Get AI disaster news
@app.route("/api/ai-disaster-news", methods=["GET"])
def get_ai_disaster_news():
    # Return sample data for now
    return jsonify([])

# Serve static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return render_template('index.html')

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)