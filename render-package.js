/**
 * This file is used to run on render.com
 * It automates the setup of PanicSense
 * Just need to use npm run build and node index.js as commands
 */
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Make sure directories exist
const dirs = ['python', 'client/dist'];
dirs.forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`Created directory: ${dir}`);
  }
});

// Copy Python files if needed
if (fs.existsSync('server/python')) {
  try {
    execSync('cp -r server/python/* python/');
    console.log('Copied Python files');
  } catch (e) {
    console.log('No Python files to copy or error copying');
  }
}

// Setup python dependencies
try {
  const pythonDeps = [
    'anthropic',
    'beautifulsoup4',
    'langdetect',
    'nltk',
    'numpy',
    'openai',
    'pandas',
    'python-dotenv',
    'requests',
    'scikit-learn',
    'torch',
    'tqdm',
    'transformers'
  ];
  execSync(`pip install ${pythonDeps.join(' ')}`);
  console.log('Installed Python dependencies');
} catch (e) {
  console.log('Error installing Python dependencies');
}

console.log('Setup complete - Ready for npm build and node index.js');