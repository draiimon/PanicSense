/**
 * Simple build script for Render deployment
 * This build script handles frontend assets but doesn't require vite to run
 */

const fs = require('fs');
const path = require('path');

// Check if dist directory exists, if not create it
const distDir = path.join(__dirname, 'dist');
if (!fs.existsSync(distDir)) {
  console.log('Creating dist directory...');
  fs.mkdirSync(distDir, { recursive: true });
}

// Create a minimal index.html file if it doesn't exist
const indexPath = path.join(distDir, 'index.html');
if (!fs.existsSync(indexPath)) {
  console.log('Creating minimal index.html...');
  const htmlContent = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PanicSense</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }
    header {
      background-color: #212529;
      color: white;
      padding: 1rem;
      text-align: center;
    }
    main {
      flex: 1;
      padding: 2rem;
      max-width: 1200px;
      margin: 0 auto;
    }
    footer {
      background-color: #f8f9fa;
      padding: 1rem;
      text-align: center;
      font-size: 0.875rem;
      border-top: 1px solid #dee2e6;
    }
    .card {
      background-color: white;
      border-radius: 4px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }
    h1, h2, h3 {
      margin-top: 0;
    }
    .api-url {
      background-color: #f8f9fa;
      padding: 0.5rem;
      border-radius: 4px;
      font-family: monospace;
    }
  </style>
</head>
<body>
  <header>
    <h1>PanicSense API Server</h1>
  </header>
  <main>
    <div class="card">
      <h2>Server Running</h2>
      <p>The PanicSense API server is running successfully. You can access the following endpoints:</p>
      
      <h3>API Endpoints</h3>
      <ul>
        <li><span class="api-url">/api/disaster-events</span> - Get all disaster events</li>
        <li><span class="api-url">/api/sentiment-posts</span> - Get sentiment analysis posts</li>
        <li><span class="api-url">/api/analyzed-files</span> - Get analyzed files</li>
        <li><span class="api-url">/api/active-upload-session</span> - Check for active upload sessions</li>
      </ul>
    </div>
  </main>
  <footer>
    <p>PanicSense Disaster Intelligence Platform &copy; 2025</p>
  </footer>
</body>
</html>
  `;
  fs.writeFileSync(indexPath, htmlContent);
}

// Create an uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  console.log('Creating uploads directory...');
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Create a python directory if it doesn't exist
const pythonDir = path.join(__dirname, 'python');
if (!fs.existsSync(pythonDir)) {
  console.log('Creating python directory...');
  fs.mkdirSync(pythonDir, { recursive: true });
}

console.log('Build completed successfully!');