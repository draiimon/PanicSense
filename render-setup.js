/**
 * PanicSense Render Setup
 * This file handles setting up the environment for Render.com deployment
 * Run before starting the server
 */
const fs = require('fs');
const { execSync } = require('child_process');
const path = require('path');

console.log('📦 Starting PanicSense Setup for Render.com');

// Essential directories
const dirs = ['python', 'client/dist', 'uploads', 'temp_files'];
dirs.forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`✅ Created directory: ${dir}`);
  }
});

// Copy Python files
try {
  if (fs.existsSync('server/python')) {
    execSync('cp -r server/python/* python/ || true');
    console.log('✅ Copied Python files to python directory');
  }
} catch (err) {
  console.log('⚠️ Error copying Python files (non-critical)');
}

// Install Python dependencies 
try {
  const dependencies = [
    'anthropic', 
    'beautifulsoup4', 
    'langdetect', 
    'nltk', 
    'numpy', 
    'openai', 
    'pandas', 
    'python-dotenv', 
    'pytz', 
    'requests', 
    'scikit-learn'
  ];
  
  console.log('🔄 Installing Python dependencies...');
  execSync(`pip install ${dependencies.join(' ')}`, { stdio: 'inherit' });
  console.log('✅ Python dependencies installed');
} catch (err) {
  console.log('⚠️ Unable to install all Python dependencies');
  console.log('🔄 Trying minimal dependencies...');
  
  try {
    execSync('pip install anthropic beautifulsoup4 numpy openai requests', { stdio: 'inherit' });
    console.log('✅ Minimal Python dependencies installed');
  } catch (e) {
    console.log('⚠️ Error installing Python dependencies');
  }
}

// Make sure Python libs are available
try {
  execSync('python -c "import nltk; nltk.download(\'punkt\')" || true');
  console.log('✅ NLTK data downloaded (or already exists)');
} catch (err) {
  console.log('⚠️ Warning: Could not download NLTK data');
}

console.log('✅ Setup complete! Ready to start the application.');