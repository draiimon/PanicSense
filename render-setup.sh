#!/bin/bash

# Ultra-simple render setup script
echo "=== 🚀 Render Setup Script ==="

# Create necessary directories
mkdir -p client/dist public uploads/{temp,data,profile_images}

# Check if public has index.html
if [ -f "public/index.html" ]; then
  echo "Found index.html in public, copying to client/dist..."
  mkdir -p client/dist
  cp -r public/* client/dist/
else
  echo "No index.html found in public, creating minimal index.html..."
  mkdir -p client/dist
  cat > client/dist/index.html << EOL
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PanicSense API</title>
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.5; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
    header { background: #e63946; color: white; padding: 1rem; border-radius: 4px; }
    h1 { margin: 0; }
    .endpoint { background: #f1faee; padding: 1rem; border-left: 4px solid #1d3557; margin: 1rem 0; }
    code { background: #e9ecef; padding: 0.2rem 0.4rem; border-radius: 3px; font-size: 0.9em; }
  </style>
</head>
<body>
  <header>
    <h1>PanicSense API Server</h1>
  </header>
  <main>
    <p>The API server is running. Access endpoints at <code>/api/*</code></p>
    <div class="endpoint">
      <h3>API Endpoints:</h3>
      <ul>
        <li><code>/api/health</code> - Health check</li>
        <li><code>/api/sentiment-posts</code> - Get sentiment data</li>
        <li><code>/api/disaster-events</code> - Get disaster events</li>
      </ul>
    </div>
  </main>
  <footer>
    <p><small>PanicSense &copy; 2025 - Disaster Intelligence Platform</small></p>
  </footer>
</body>
</html>
EOL
fi

echo "=== ✅ Render setup complete ==="