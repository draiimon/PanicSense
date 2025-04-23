/**
 * DIRECT PRODUCTION BUILD SCRIPT
 * Hindi gumagamit ng Vite - pang-Render Free Tier ito!
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🚀 PANICSENSE PRODUCTION BUILD - NO VITE NEEDED!');

// Install only production dependencies
try {
  console.log('📦 Installing dependencies...');
  execSync('npm install --omit=dev', { stdio: 'inherit' });
} catch (error) {
  console.error('Error installing dependencies, pero tuloy pa rin!');
}

// Create folders
console.log('📁 Creating necessary folders...');
fs.mkdirSync('dist', { recursive: true });
fs.mkdirSync('dist/public', { recursive: true });
fs.mkdirSync('uploads/temp', { recursive: true });
fs.mkdirSync('uploads/data', { recursive: true });
fs.mkdirSync('uploads/profile_images', { recursive: true });
fs.mkdirSync('python', { recursive: true });

// Copy server files
console.log('📋 Copying server files...');
fs.cpSync('server', 'dist', { recursive: true });

// Create a simple index.html for API server
console.log('🌐 Creating API server page...');
const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>PanicSense API Server</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
    h1 { color: #333; }
    .status { background: #f0f0f0; padding: 10px; border-radius: 4px; }
    .version { margin-top: 20px; font-size: 0.8em; color: #666; }
  </style>
</head>
<body>
  <h1>✅ PanicSense API Server</h1>
  <p>PanicSense API server is running. Frontend not built during deployment to save resources.</p>
  <div class="status">
    <p><strong>Status:</strong> Running</p>
    <p><strong>Deployment:</strong> Render Free Tier</p>
    <p><strong>Server Type:</strong> API Only</p>
  </div>
  <div class="version">
    <p>Server Version: ${new Date().toISOString()}</p>
  </div>
</body>
</html>`;
fs.writeFileSync('dist/public/index.html', html);

// Copy Python files
if (fs.existsSync('server/python')) {
  console.log('🐍 Copying Python files...');
  fs.cpSync('server/python', 'python', { recursive: true });
}

// Create Python requirements
console.log('🐍 Setting up Python requirements...');
const requirements = `anthropic
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
snscrape
tqdm`;
fs.writeFileSync('requirements.txt', requirements);

// Success message
console.log('✅ BUILD COMPLETED! Now ready for Render (no need for Vite)');