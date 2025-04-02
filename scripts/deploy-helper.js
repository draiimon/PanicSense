/**
 * Deployment Helper for PanicSense
 * 
 * This script helps with deployment preparation for both Replit and Render platforms.
 * It ensures consistent environment setup and validates required configurations.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Detect platform
const isRender = Boolean(process.env.RENDER);
const isReplit = Boolean(process.env.REPL_ID);
const platform = isRender ? 'Render' : isReplit ? 'Replit' : 'Local';

console.log(`Preparing deployment for ${platform} environment...`);

// Validate database connection
function validateDatabase() {
  if (!process.env.DATABASE_URL) {
    console.error('ERROR: DATABASE_URL environment variable is not set.');
    console.error('Please set it in your environment or .env file.');
    process.exit(1);
  }
  
  console.log('✓ Database URL is configured.');
}

// Check for required Python packages
function validatePythonSetup() {
  try {
    const requiredPackages = ['nltk', 'scikit-learn', 'numpy', 'pandas', 'langdetect'];
    
    for (const pkg of requiredPackages) {
      try {
        execSync(`python3 -c "import ${pkg}"`, { stdio: 'ignore' });
        console.log(`✓ Python package ${pkg} is installed.`);
      } catch (e) {
        console.warn(`⚠ Python package ${pkg} is not installed. Installing...`);
        execSync(`pip install ${pkg}`, { stdio: 'inherit' });
      }
    }
  } catch (e) {
    console.warn('⚠ Could not verify Python packages. Make sure Python 3.x is installed with required packages.');
  }
}

// Ensure build directory exists
function prepareBuildDirectory() {
  const distDir = path.join(__dirname, '..', 'dist');
  const publicDir = path.join(distDir, 'public');
  
  if (!fs.existsSync(distDir)) {
    fs.mkdirSync(distDir, { recursive: true });
    console.log('✓ Created dist directory.');
  }
  
  if (!fs.existsSync(publicDir)) {
    fs.mkdirSync(publicDir, { recursive: true });
    console.log('✓ Created public directory.');
  }
}

// Ensure temp directories exist
function prepareTempDirectories() {
  const tempDir = path.join(require('os').tmpdir(), 'disaster-sentiment');
  
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
    console.log('✓ Created temp directory for file processing.');
  }
  
  // For Linux systems, also create a RAM-based tmpfs directory for faster processing
  if (process.platform === 'linux') {
    try {
      if (fs.existsSync('/dev/shm')) {
        const ramTempDir = path.join('/dev/shm', 'disaster-sentiment');
        if (!fs.existsSync(ramTempDir)) {
          fs.mkdirSync(ramTempDir, { recursive: true });
          console.log('✓ Created RAM-based temp directory for faster processing.');
        }
      }
    } catch (e) {
      console.warn('⚠ Could not create RAM-based temp directory.');
    }
  }
}

// Render-specific preparations
function prepareForRender() {
  if (!isRender) return;
  
  console.log('Performing Render-specific setup...');
  
  // Ensure NODE_ENV is set to production
  process.env.NODE_ENV = 'production';
  
  // Convert port to number if present
  if (process.env.PORT) {
    process.env.PORT = parseInt(process.env.PORT, 10).toString();
  }
}

// Replit-specific preparations
function prepareForReplit() {
  if (!isReplit) return;
  
  console.log('Performing Replit-specific setup...');
  
  // Create symbolic link for assets if needed
  const assetsSourceDir = path.join(__dirname, '..', 'attached_assets');
  const assetsTargetDir = path.join(__dirname, '..', 'assets');
  
  if (fs.existsSync(assetsSourceDir) && !fs.existsSync(assetsTargetDir)) {
    try {
      fs.symlinkSync(assetsSourceDir, assetsTargetDir, 'dir');
      console.log('✓ Created symbolic link for assets directory.');
    } catch (e) {
      console.warn('⚠ Could not create symbolic link for assets. Using copy instead.');
      // If symlink fails, try copy
      if (!fs.existsSync(assetsTargetDir)) {
        fs.mkdirSync(assetsTargetDir, { recursive: true });
      }
      
      const files = fs.readdirSync(assetsSourceDir);
      for (const file of files) {
        const sourcePath = path.join(assetsSourceDir, file);
        const targetPath = path.join(assetsTargetDir, file);
        fs.copyFileSync(sourcePath, targetPath);
      }
      console.log('✓ Copied assets to target directory.');
    }
  }
}

// Run all checks
function runDeploymentChecks() {
  validateDatabase();
  validatePythonSetup();
  prepareBuildDirectory();
  prepareTempDirectories();
  prepareForRender();
  prepareForReplit();
  
  console.log('✓ Deployment preparation completed successfully!');
}

runDeploymentChecks();