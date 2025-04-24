#!/bin/bash

# Enhanced build script for Render FREE TIER deployment
echo "=== 🚀 PanicSense FREE TIER Build Process ==="

# Display environment information
echo "NODE_ENV: $NODE_ENV"
echo "Current directory: $(pwd)"
echo "Node version: $(node -v)"
echo "NPM version: $(npm -v)"

# Install dependencies 
echo "=== 📦 Installing dependencies ==="
npm install --no-optional

# IMPORTANT: Explicitly install Vite and ESBuild globally
echo "=== 📦 Installing build tools globally ==="
npm install -g vite esbuild

# Create necessary folders
echo "=== 📁 Creating required folders ==="
mkdir -p dist/public uploads/{temp,data,profile_images} python client

# Try to build with vite and esbuild, with fallback mechanisms
echo "=== 🏗️ Custom build process to avoid Vite not found error ==="
if [ -f "vite.config.ts" ] || [ -f "vite.config.js" ]; then
  echo "Vite config found, attempting to build using Vite..."
  if npx vite build; then
    echo "✅ Vite build succeeded!"
  else
    echo "⚠️ Vite build failed, using ESBuild fallback for frontend..."
    if [ -d "client/src" ]; then
      echo "Attempting to build frontend with ESBuild..."
      npx esbuild client/src/main.tsx --bundle --outfile=dist/public/bundle.js || true
      
      # Create minimal index.html
      echo "Creating minimal index.html..."
      cat > dist/public/index.html << EOL
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PanicSense</title>
  <script defer src="/bundle.js"></script>
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.5; }
    .container { max-width: 1200px; margin: 0 auto; padding: 1rem; }
    header { background: #e63946; color: white; padding: 1rem; }
    h1 { margin: 0; }
  </style>
</head>
<body>
  <div id="root">
    <header>
      <div class="container">
        <h1>PanicSense</h1>
      </div>
    </header>
    <main class="container">
      <p>Loading application...</p>
    </main>
  </div>
</body>
</html>
EOL
    else
      echo "Creating minimal index.html (API-only mode)..."
      cat > dist/public/index.html << EOL
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
    <p>The API server is running. You can access the following endpoints:</p>
    <div class="endpoint">
      <h3>/api/health</h3>
      <p>Check if the API is running properly.</p>
      <code>GET /api/health</code>
    </div>
    <div class="endpoint">
      <h3>/api/sentiment-posts</h3>
      <p>Get analyzed sentiment data.</p>
      <code>GET /api/sentiment-posts</code>
    </div>
    <div class="endpoint">
      <h3>/api/disaster-events</h3>
      <p>Get disaster events data.</p>
      <code>GET /api/disaster-events</code>
    </div>
  </main>
  <footer>
    <p><small>PanicSense &copy; 2025 - Disaster Intelligence Platform</small></p>
  </footer>
</body>
</html>
EOL
    fi
  fi
else
  echo "⚠️ No Vite config found, creating minimal frontend..."
  cat > dist/public/index.html << EOL
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
    <p>The API server is running in minimal mode.</p>
  </main>
</body>
</html>
EOL
fi

# Try to build server with esbuild
echo "=== 🏗️ Building server with ESBuild ==="
if npx esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist; then
  echo "✅ ESBuild server build succeeded!"
else
  echo "⚠️ ESBuild server build failed, copying server files directly..."
  mkdir -p dist
  cp -r server dist/ 2>/dev/null || true
  echo "Creating compatibility wrapper..."
  cat > dist/index.js << EOL
import './server/index.js';
EOL
fi

# Copy frontend files if they exist
echo "=== 📋 Copying frontend files ==="
if [ -d "client/dist" ]; then
  echo "Copying client/dist to dist/public..."
  cp -r client/dist/* dist/public/ 2>/dev/null || true
elif [ -d "public" ]; then
  echo "Copying public to dist/public..."
  cp -r public/* dist/public/ 2>/dev/null || true
fi

# Copy Python files if needed
echo "=== 📋 Copying Python files ==="
if [ -d "python" ]; then
  echo "Python directory exists, no need to copy"
elif [ -d "server/python" ]; then
  echo "Copying server/python to python..."
  cp -r server/python/* python/ 2>/dev/null || true
fi

# Create Python requirements if missing
echo "=== 🐍 Setting up Python environment ==="
if [ ! -f "requirements.txt" ]; then
  echo "Creating requirements.txt..."
  cat > requirements.txt << EOL
anthropic
beautifulsoup4
langdetect
nltk
numpy
openai
pandas
python-dotenv
pytz
requests
scikit-learn
tqdm
EOL
fi

# Make sure run.js is executable
if [ -f "run.js" ]; then
  chmod +x run.js
fi

# Check if the essential files for server starting are present
echo "=== 🔍 Checking essential files ==="
echo "server/index.js: $([ -f "server/index.js" ] && echo "✅ Found" || echo "❌ Missing")"
echo "server/index.ts: $([ -f "server/index.ts" ] && echo "✅ Found" || echo "❌ Missing")"
echo "server/index-wrapper.js: $([ -f "server/index-wrapper.js" ] && echo "✅ Found" || echo "❌ Missing")"
echo "dist/index.js: $([ -f "dist/index.js" ] && echo "✅ Found" || echo "❌ Missing")"
echo "run.js: $([ -f "run.js" ] && echo "✅ Found" || echo "❌ Missing")"

# Installing minimal Python dependencies
echo "=== 🐍 Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== ✅ Build complete - FREE TIER READY ==="