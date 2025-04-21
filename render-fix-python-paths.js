/**
 * RENDER PYTHON PATH FIX UTILITY
 * 
 * This script helps ensure Python scripts are properly located and configured for Render deployment.
 * Run this script as part of the build process when deploying to Render.
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

// Get directory of this script
const scriptDir = process.cwd();

// Critical paths to check and create
const criticalPaths = {
  // Python directories
  pythonRoot: path.join(scriptDir, 'python'),
  serverPython: path.join(scriptDir, 'server', 'python'),
  
  // Python scripts
  processScript: path.join(scriptDir, 'server', 'python', 'process.py'),
  alternateProcess: path.join(scriptDir, 'python', 'process.py'),
  
  // Temporary directories
  tempDir: path.join('/tmp', 'disaster-sentiment'),
  
  // Log directories
  logsDir: path.join(scriptDir, 'logs')
};

console.log('============================================');
console.log('🐍 PANICSENSE PYTHON PATH FIXER FOR RENDER');
console.log('============================================');
console.log(`Running from: ${scriptDir}`);

// Check Python installation
try {
  const pythonVersion = execSync('python3 --version').toString().trim();
  console.log(`🐍 Python detected: ${pythonVersion}`);
  
  // Check pip
  try {
    const pipVersion = execSync('pip3 --version').toString().trim();
    console.log(`📦 Pip detected: ${pipVersion}`);
  } catch (error) {
    console.warn('⚠️ Could not detect pip:', error.message);
  }
} catch (error) {
  console.error('❌ Python not found:', error.message);
}

// Create missing directories
console.log('\n=== Creating critical directories ===');
Object.entries(criticalPaths).forEach(([name, dirPath]) => {
  if (name.endsWith('Dir') || name.includes('python')) {
    try {
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
        console.log(`✅ Created ${name}: ${dirPath}`);
      } else {
        console.log(`✅ ${name} already exists: ${dirPath}`);
      }
    } catch (error) {
      console.error(`❌ Error creating ${name}:`, error.message);
    }
  }
});

// Check and fix Python scripts
console.log('\n=== Checking Python scripts ===');

// First, look for process.py in server/python
const sourcePaths = [
  path.join(scriptDir, 'server', 'python', 'process.py'),
  path.join(scriptDir, 'python', 'process.py')
];

let processScript = null;

// Find first existing script
for (const sourcePath of sourcePaths) {
  if (fs.existsSync(sourcePath)) {
    processScript = sourcePath;
    console.log(`✅ Found process.py at: ${sourcePath}`);
    break;
  }
}

// If process.py was found, copy it to both locations
if (processScript) {
  try {
    // Copy to root python directory
    fs.copyFileSync(processScript, path.join(scriptDir, 'python', 'process.py'));
    console.log(`✅ Copied process.py to ${path.join(scriptDir, 'python')}`);
    
    // Copy to server/python directory
    fs.copyFileSync(processScript, path.join(scriptDir, 'server', 'python', 'process.py'));
    console.log(`✅ Copied process.py to ${path.join(scriptDir, 'server', 'python')}`);
  } catch (error) {
    console.error('❌ Error copying process.py:', error.message);
  }
} else {
  console.error('❌ Could not find process.py in any known location!');
  
  // List directory contents to help diagnose
  try {
    console.log('\n=== Directory contents ===');
    
    const serverDir = path.join(scriptDir, 'server');
    if (fs.existsSync(serverDir)) {
      console.log(`📂 Contents of ${serverDir}:`);
      console.log(fs.readdirSync(serverDir));
      
      const serverPythonDir = path.join(serverDir, 'python');
      if (fs.existsSync(serverPythonDir)) {
        console.log(`📂 Contents of ${serverPythonDir}:`);
        console.log(fs.readdirSync(serverPythonDir));
      }
    }
    
    const rootPythonDir = path.join(scriptDir, 'python');
    if (fs.existsSync(rootPythonDir)) {
      console.log(`📂 Contents of ${rootPythonDir}:`);
      console.log(fs.readdirSync(rootPythonDir));
    }
  } catch (error) {
    console.error('❌ Error listing directories:', error.message);
  }
}

// Check Python dependencies (optional)
console.log('\n=== Checking Python dependencies ===');
const requiredPackages = ['pandas', 'numpy', 'nltk', 'scikit-learn'];

try {
  const installedPackages = execSync('pip3 list').toString();
  
  requiredPackages.forEach(pkg => {
    if (installedPackages.includes(pkg)) {
      console.log(`✅ ${pkg} is installed`);
    } else {
      console.warn(`⚠️ ${pkg} not found - may need to install`);
    }
  });
} catch (error) {
  console.error('❌ Could not check Python packages:', error.message);
}

console.log('\n============================================');
console.log('✅ Python path fix process complete');
console.log('============================================');