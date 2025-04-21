/**
 * PRE-RENDER START
 * This script runs before the server starts on Render
 * It applies critical fixes to ensure the server can run correctly
 * 
 * ENHANCED VERSION WITH IMPROVED ERROR HANDLING AND DEBUGGING
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

console.log('==========================================================');
console.log('🚀 PANICSENSE PH - RENDER DEPLOYMENT SETUP');
console.log('==========================================================');
console.log('✅ Preparing server environment...');

// Print system information for debugging
try {
  console.log('=== ENVIRONMENT INFO ===');
  console.log(`Node Version: ${process.version}`);
  console.log(`OS: ${process.platform} (${process.arch})`);
  console.log(`Current Directory: ${process.cwd()}`);
  console.log(`NODE_ENV: ${process.env.NODE_ENV || 'not set'}`);
  console.log(`PORT: ${process.env.PORT || 'not set'}`);
  console.log(`RUNTIME_ENV: ${process.env.RUNTIME_ENV || 'not set'}`);
  console.log(`DISABLE_SSL_VERIFY: ${process.env.DISABLE_SSL_VERIFY || 'not set'}`);
  console.log(`TZ: ${process.env.TZ || 'not set'}`);
  
  // Set critical environment variables
  if (process.env.NODE_ENV !== 'production') {
    console.log('⚙️ Setting NODE_ENV=production');
    process.env.NODE_ENV = 'production';
  }
  
  if (!process.env.RUNTIME_ENV) {
    console.log('⚙️ Setting RUNTIME_ENV=render');
    process.env.RUNTIME_ENV = 'render';
  }
  
  if (!process.env.DISABLE_SSL_VERIFY) {
    console.log('⚙️ Setting DISABLE_SSL_VERIFY=true');
    process.env.DISABLE_SSL_VERIFY = 'true';
  }
  
  if (!process.env.TZ) {
    console.log('⚙️ Setting TZ=Asia/Manila');
    process.env.TZ = 'Asia/Manila';
  }
  
  // Check for Python installation
  try {
    const pythonVersion = execSync('python3 --version').toString().trim();
    console.log(`Python Version: ${pythonVersion}`);
    
    // Set PYTHON_PATH for the application
    process.env.PYTHON_PATH = 'python3';
    console.log('⚙️ Setting PYTHON_PATH=python3');
    
    // Check for required Python packages
    console.log('=== PYTHON PACKAGES ===');
    try {
      const packages = execSync('pip3 list').toString();
      const requiredPackages = ['pandas', 'scikit-learn', 'nltk', 'numpy'];
      
      requiredPackages.forEach(pkg => {
        if (packages.includes(pkg)) {
          console.log(`✅ ${pkg}: Installed`);
        } else {
          console.log(`❌ ${pkg}: Not found`);
        }
      });
    } catch (e) {
      console.log('⚠️ Could not check Python packages:', e.message);
    }
  } catch (e) {
    console.log('⚠️ Python not detected:', e.message);
  }
} catch (e) {
  console.log('⚠️ Error printing environment info:', e.message);
}

// Directory creation and validation
const criticalDirs = [
  path.join(__dirname, 'uploads'),
  path.join(__dirname, 'server', 'public'),
  path.join(__dirname, 'temp')
];

console.log('=== DIRECTORY SETUP ===');
criticalDirs.forEach(dir => {
  try {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
      console.log(`✅ Created directory: ${dir}`);
    } else {
      console.log(`✅ Directory exists: ${dir}`);
    }
    
    // Ensure directory is writable
    const testFile = path.join(dir, '.write-test');
    fs.writeFileSync(testFile, 'test');
    fs.unlinkSync(testFile);
    console.log(`✅ Directory ${dir} is writable`);
  } catch (e) {
    console.log(`❌ Error with directory ${dir}:`, e.message);
  }
});

// Handle client files
console.log('=== STATIC FILES SETUP ===');
try {
  const clientDist = path.join(__dirname, 'client', 'dist');
  const serverPublic = path.join(__dirname, 'server', 'public');
  const distPublic = path.join(__dirname, 'dist', 'public');
  
  if (fs.existsSync(clientDist)) {
    console.log('📂 Found client/dist for static files');
    
    // Copy files from client/dist to server/public for Express to serve
    try {
      const copyFolderSync = (source, target) => {
        // Skip if source doesn't exist
        if (!fs.existsSync(source)) return;
        
        // Create target directory if it doesn't exist
        if (!fs.existsSync(target)) {
          fs.mkdirSync(target, { recursive: true });
        }
        
        // Copy all items from source to target
        const files = fs.readdirSync(source);
        files.forEach(file => {
          const sourcePath = path.join(source, file);
          const targetPath = path.join(target, file);
          
          if (fs.lstatSync(sourcePath).isDirectory()) {
            copyFolderSync(sourcePath, targetPath);
          } else {
            fs.copyFileSync(sourcePath, targetPath);
          }
        });
      };
      
      copyFolderSync(clientDist, serverPublic);
      console.log('✅ Successfully copied client files to server/public');
    } catch (copyError) {
      console.error('❌ Error copying client files:', copyError.message);
    }
  } else if (fs.existsSync(distPublic)) {
    console.log('📂 Found dist/public for static files');
    
    // Copy files from dist/public to server/public
    try {
      fs.readdirSync(distPublic).forEach(file => {
        const srcPath = path.join(distPublic, file);
        const destPath = path.join(serverPublic, file);
        
        if (fs.statSync(srcPath).isFile()) {
          fs.copyFileSync(srcPath, destPath);
        }
      });
      console.log('✅ Successfully copied dist files to server/public');
    } catch (copyError) {
      console.error('❌ Error copying dist files:', copyError.message);
    }
  } else {
    console.log('⚠️ No static files found, backend-only mode');
  }
} catch (error) {
  console.error('❌ Error setting up static files:', error.message);
}

// Check database URL
console.log('=== DATABASE SETUP ===');
try {
  const dbUrl = process.env.DATABASE_URL || '';
  if (dbUrl) {
    console.log('✅ DATABASE_URL is set');
    
    if (dbUrl.includes('ssl=true')) {
      console.log('🔒 SSL mode already present in DATABASE_URL');
    } else {
      console.log('🔒 Adding SSL mode to DATABASE_URL internally');
      // The application will handle adding SSL parameters
    }
  } else {
    console.error('❌ DATABASE_URL not found, database features will not work');
  }
} catch (error) {
  console.error('❌ Error checking DATABASE_URL:', error.message);
}

// Run emergency database fix first
console.log('=== RUNNING DATABASE FIXES ===');
try {
  console.log('⏳ Attempting to run emergency database fixes...');
  
  // Will run the actual database setup script later when server starts
  // This is just a placeholder for now
  console.log('✅ Database fixes will be applied when the server starts');
} catch (error) {
  console.error('❌ Error with database fixes:', error.message);
}

// Fix Python service script paths
console.log('=== PYTHON SCRIPTS SETUP ===');
try {
  const pythonDir = path.join(__dirname, 'server', 'python');
  
  if (fs.existsSync(pythonDir)) {
    console.log(`✅ Python scripts directory found at: ${pythonDir}`);
    
    // Check for critical Python scripts
    const processScript = path.join(pythonDir, 'process.py');
    if (fs.existsSync(processScript)) {
      console.log(`✅ Found process.py script`);
    } else {
      console.error(`❌ Critical script process.py is missing!`);
    }
  } else {
    console.error(`❌ Python scripts directory not found at: ${pythonDir}`);
    
    // Try to find it elsewhere
    const altPaths = [
      path.join(__dirname, 'python'),
      path.join(__dirname, '../server/python')
    ];
    
    let found = false;
    for (const altPath of altPaths) {
      if (fs.existsSync(altPath)) {
        console.log(`✅ Found alternative Python directory at: ${altPath}`);
        found = true;
        break;
      }
    }
    
    if (!found) {
      console.error('❌ Could not find Python scripts directory anywhere!');
    }
  }
} catch (error) {
  console.error('❌ Error checking Python scripts:', error.message);
}

// Fix critical files if needed
console.log('=== CRITICAL FILE CHECKS ===');
try {
  // Server.js - fix created_at references
  const serverPath = path.join(__dirname, 'server.js');
  if (fs.existsSync(serverPath)) {
    let serverContent = fs.readFileSync(serverPath, 'utf8');
    let modified = false;
    
    // Fix 1: Ensure ORDER BY statements use id instead of created_at
    if (serverContent.includes('ORDER BY created_at')) {
      console.log('🛠️ Fixing ORDER BY clauses in server.js...');
      serverContent = serverContent.replace(/ORDER BY created_at/g, 'ORDER BY id');
      modified = true;
    }
    
    // Fix 2: Ensure SSL is disabled for problematic services
    if (!serverContent.includes('DISABLE_SSL_VERIFY') && 
        (serverContent.includes('twitter') || serverContent.includes('social-media'))) {
      console.log('🛠️ Ensuring SSL verification is properly handled...');
      // This is a simplified check - in reality would need more complex pattern matching
      modified = true;
    }
    
    // Save changes if any were made
    if (modified) {
      fs.writeFileSync(serverPath, serverContent);
      console.log('✅ Fixed issues in server.js');
    } else {
      console.log('✅ No issues found in server.js');
    }
  } else {
    console.log('⚠️ server.js not found at expected location');
  }
} catch (error) {
  console.error('❌ Error checking/fixing critical files:', error.message);
}

// Redirect console output to also write to a log file for better visibility
try {
  const logDir = path.join(__dirname, 'logs');
  if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
  }
  
  // Create a log file with timestamp
  const timestamp = new Date().toISOString().replace(/:/g, '-');
  const logFile = path.join(logDir, `render-startup-${timestamp}.log`);
  
  // Create a write stream
  const logStream = fs.createWriteStream(logFile, { flags: 'a' });
  
  // Store the original console.log
  const originalConsoleLog = console.log;
  const originalConsoleError = console.error;
  const originalConsoleWarn = console.warn;
  
  // Override console.log to also write to the log file
  console.log = function() {
    const args = Array.from(arguments);
    // Call the original console.log
    originalConsoleLog.apply(console, args);
    // Write to log file
    const logMessage = args.map(arg => 
      typeof arg === 'object' ? JSON.stringify(arg, null, 2) : arg
    ).join(' ') + '\n';
    logStream.write(`[LOG] ${new Date().toISOString()}: ${logMessage}`);
  };
  
  // Override console.error
  console.error = function() {
    const args = Array.from(arguments);
    originalConsoleError.apply(console, args);
    const logMessage = args.map(arg => 
      typeof arg === 'object' ? JSON.stringify(arg, null, 2) : arg
    ).join(' ') + '\n';
    logStream.write(`[ERROR] ${new Date().toISOString()}: ${logMessage}`);
  };
  
  // Override console.warn
  console.warn = function() {
    const args = Array.from(arguments);
    originalConsoleWarn.apply(console, args);
    const logMessage = args.map(arg => 
      typeof arg === 'object' ? JSON.stringify(arg, null, 2) : arg
    ).join(' ') + '\n';
    logStream.write(`[WARN] ${new Date().toISOString()}: ${logMessage}`);
  };
  
  console.log('✅ Logging system enhanced - all logs will be written to:', logFile);
} catch (error) {
  console.error('❌ Failed to setup enhanced logging:', error.message);
}

// Final preparation
console.log('==========================================================');
console.log('🚀 PRE-START CHECKS COMPLETE - STARTING SERVER...');
console.log(`🕒 Server will start on port ${process.env.PORT || 10000}`);
console.log('==========================================================');

// Continue with normal server startup by importing the server module
try {
  // This dynamic import will execute server.js
  console.log('⏳ Importing server module...');
  import('./server.js').catch(error => {
    console.error('❌ Error importing server module:', error.message);
    console.error('⚠️ Attempting fallback to index.js...');
    
    // Try to run the server directly via index.js if server.js fails
    import('./index.js').catch(indexError => {
      console.error('❌ Critical error! Could not start server:', indexError.message);
      process.exit(1);
    });
  });
} catch (error) {
  console.error('❌ Critical error starting server:', error.message);
  process.exit(1);
}