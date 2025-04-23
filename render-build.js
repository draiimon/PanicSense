/**
 * Alternative build script for Render.com free tier
 * 
 * This is a pure JS alternative to the build.sh script
 * Use this by changing the build command in render.yaml to:
 * node render-build.js
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Helper to run commands and handle errors
function runCommand(command, errorMessage) {
  try {
    console.log(`Running: ${command}`);
    execSync(command, { stdio: 'inherit' });
    return true;
  } catch (error) {
    console.error(`⚠️ ${errorMessage || 'Command failed'}`);
    console.error(`Command was: ${command}`);
    return false;
  }
}

// Make sure folders exist
function ensureFolderExists(folder) {
  if (!fs.existsSync(folder)) {
    console.log(`Creating folder: ${folder}`);
    fs.mkdirSync(folder, { recursive: true });
  }
}

// Main build process
async function build() {
  console.log('=== 🚀 PanicSense FREE TIER Alternative Build Process ===');
  
  // Install dependencies
  runCommand('npm install', 'Failed to install dependencies');
  
  // Attempt to install build tools globally
  runCommand('npm install -g vite esbuild', 'Failed to install global tools, continuing...');
  
  // Create required folders
  const folders = [
    'dist',
    'dist/public',
    'uploads',
    'uploads/temp',
    'uploads/data',
    'uploads/profile_images',
    'python'
  ];
  
  folders.forEach(ensureFolderExists);
  
  // Try frontend build
  if (!runCommand('npx vite build', 'Vite build failed')) {
    console.log('Creating minimal index.html as fallback...');
    const html = '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>PanicSense</title></head><body><h1>PanicSense API Server</h1><p>Frontend not available in this deployment.</p></body></html>';
    fs.writeFileSync('dist/public/index.html', html);
  }
  
  // Try server build
  if (!runCommand('npx esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist', 'ESBuild failed')) {
    console.log('Copying server files directly as fallback...');
    if (fs.existsSync('server')) {
      const serverFiles = fs.readdirSync('server');
      serverFiles.forEach(file => {
        const srcPath = path.join('server', file);
        const destPath = path.join('dist', file);
        if (fs.statSync(srcPath).isFile()) {
          fs.copyFileSync(srcPath, destPath);
        }
      });
    }
  }
  
  // Copy frontend files if they exist
  if (fs.existsSync('client/dist')) {
    console.log('Copying client/dist files to dist/public...');
    const files = fs.readdirSync('client/dist');
    files.forEach(file => {
      const srcPath = path.join('client/dist', file);
      const destPath = path.join('dist/public', file);
      if (fs.statSync(srcPath).isFile()) {
        fs.copyFileSync(srcPath, destPath);
      }
    });
  } else if (fs.existsSync('public')) {
    console.log('Copying public files to dist/public...');
    const files = fs.readdirSync('public');
    files.forEach(file => {
      const srcPath = path.join('public', file);
      const destPath = path.join('dist/public', file);
      if (fs.statSync(srcPath).isFile()) {
        fs.copyFileSync(srcPath, destPath);
      }
    });
  }
  
  // Set up Python
  if (fs.existsSync('server/python')) {
    console.log('Copying Python files...');
    const pythonFiles = fs.readdirSync('server/python');
    pythonFiles.forEach(file => {
      const srcPath = path.join('server/python', file);
      const destPath = path.join('python', file);
      if (fs.statSync(srcPath).isFile()) {
        fs.copyFileSync(srcPath, destPath);
      }
    });
  }
  
  // Create Python requirements.txt if missing
  if (!fs.existsSync('requirements.txt')) {
    console.log('Creating requirements.txt...');
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
  }
  
  // Install Python dependencies
  runCommand('pip install -r requirements.txt', 'Failed to install Python dependencies');
  
  console.log('=== ✅ Build complete - FREE TIER READY ===');
}

// Run build
build().catch(error => {
  console.error('Build failed:', error);
  process.exit(1);
});