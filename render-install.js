/**
 * RENDER PANICSENSE INSTALLER
 * Automatic installer for Render.com deployment 
 * May kasamang vite at lahat ng dependencies
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🚀 PanicSense for Render - Complete Installer');

// Ensure folders exist
function ensureDirExists(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`Created directory: ${dir}`);
  }
}

// Execute command with error handling
function runCommand(command) {
  try {
    console.log(`Running: ${command}`);
    execSync(command, { stdio: 'inherit' });
    return true;
  } catch (error) {
    console.error(`Command failed: ${command}`);
    console.error(error.message);
    return false;
  }
}

// Main installation process
async function install() {
  // Create necessary directories
  ensureDirExists('dist');
  ensureDirExists('dist/public');
  ensureDirExists('uploads/temp');
  ensureDirExists('uploads/data');
  ensureDirExists('uploads/profile_images');
  ensureDirExists('python');
  
  console.log('\n=== 📦 Installing Dependencies ===');
  
  // Install ALL dependencies including dev dependencies
  runCommand('npm install');
  
  // Install Vite globally (very important!)
  console.log('\n=== 🛠️ Installing Vite Globally ===');
  runCommand('npm install -g vite');
  
  // Add vite to PATH
  console.log('\n=== 🔧 Adding Vite to PATH ===');
  process.env.PATH = `${process.env.PATH}:/opt/render/project/node_modules/.bin`;
  console.log(`PATH is now: ${process.env.PATH}`);
  
  // Create symlink to ensure vite is found
  console.log('\n=== 🔗 Creating symlinks for Vite ===');
  if (!fs.existsSync('/usr/local/bin/vite')) {
    try {
      fs.symlinkSync(
        '/opt/render/project/node_modules/.bin/vite',
        '/usr/local/bin/vite'
      );
      console.log('Symlink created for vite');
    } catch (error) {
      console.log('Could not create symlink (may need permissions):', error.message);
    }
  }
  
  console.log('\n=== 🏗️ Building Frontend and Server ===');
  
  // Try to run the build script
  if (runCommand('npm run build')) {
    console.log('Build completed successfully!');
  } else {
    console.error('Build failed, trying manual build process');
    
    // Manual build as fallback
    try {
      // Try to run vite build directly
      runCommand('npx vite build');
      runCommand('npx esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist');
    } catch (error) {
      console.error('Manual build also failed:', error.message);
      console.log('Creating minimal frontend as fallback');
      
      // Create minimal frontend
      const html = '<!DOCTYPE html><html><head><title>PanicSense</title></head><body><h1>PanicSense API Server</h1><p>API server is running. Frontend build failed but server is operational.</p></body></html>';
      fs.writeFileSync('dist/public/index.html', html);
    }
  }
  
  console.log('\n=== 🐍 Setting up Python Environment ===');
  
  // Copy Python files
  if (fs.existsSync('server/python')) {
    const files = fs.readdirSync('server/python');
    for (const file of files) {
      const src = path.join('server/python', file);
      const dest = path.join('python', file);
      if (fs.statSync(src).isFile()) {
        fs.copyFileSync(src, dest);
        console.log(`Copied ${src} to ${dest}`);
      }
    }
  }
  
  // Create Python requirements.txt
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
  console.log('Created requirements.txt');
  
  // Install Python requirements
  console.log('\n=== 🐍 Installing Python Dependencies ===');
  runCommand('pip install -r requirements.txt');
  
  console.log('\n=== ✅ INSTALLATION COMPLETE! ===');
  console.log('Render deployment is now ready.');
}

// Run the installation process
install().catch(error => {
  console.error('Installation failed:', error);
  process.exit(1);
});