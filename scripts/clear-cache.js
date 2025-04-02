/**
 * Cache Cleaning Script for PanicSense
 * 
 * This script cleans temporary files and caches to optimize performance
 * and reduce disk usage in both development and production environments.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync } = require('child_process');

const isReplit = Boolean(process.env.REPL_ID);
const isRender = Boolean(process.env.RENDER);
const platform = isRender ? 'Render' : isReplit ? 'Replit' : 'Local';

console.log(`Running cache cleaning on ${platform} platform`);

// Clean temporary files
function cleanTempFiles() {
  console.log('\n=== CLEANING TEMPORARY FILES ===');
  
  // Determine temp directory location based on platform
  const tempDir = process.env.TEMP_DIR || 
                 (isRender || isReplit ? '/tmp/disaster-sentiment' : 
                 path.join(os.tmpdir(), 'disaster-sentiment'));
  
  if (fs.existsSync(tempDir)) {
    try {
      // Read directory contents
      const files = fs.readdirSync(tempDir);
      console.log(`Found ${files.length} files in temp directory`);
      
      // Delete all files except directories
      let deletedCount = 0;
      for (const file of files) {
        const filePath = path.join(tempDir, file);
        
        try {
          const stats = fs.statSync(filePath);
          if (stats.isFile()) {
            fs.unlinkSync(filePath);
            deletedCount++;
          }
        } catch (e) {
          console.error(`Error processing file ${file}:`, e.message);
        }
      }
      
      console.log(`✅ Deleted ${deletedCount} temporary files`);
    } catch (error) {
      console.error('❌ Error cleaning temp directory:', error.message);
    }
  } else {
    console.log(`Temp directory ${tempDir} does not exist, creating it`);
    fs.mkdirSync(tempDir, { recursive: true });
  }
}

// Clean Node.js caches
function cleanNodeCaches() {
  console.log('\n=== CLEANING NODE.JS CACHES ===');
  
  // Clean node_modules/.cache if it exists
  const nodeCacheDir = path.join(process.cwd(), 'node_modules', '.cache');
  if (fs.existsSync(nodeCacheDir)) {
    try {
      // Use rimraf-like approach for deleting directory contents
      function deleteFolderRecursive(dirPath) {
        if (fs.existsSync(dirPath)) {
          fs.readdirSync(dirPath).forEach((file) => {
            const curPath = path.join(dirPath, file);
            if (fs.lstatSync(curPath).isDirectory()) {
              deleteFolderRecursive(curPath);
            } else {
              fs.unlinkSync(curPath);
            }
          });
          
          // Don't delete the .cache directory itself, just its contents
          if (dirPath !== nodeCacheDir) {
            fs.rmdirSync(dirPath);
          }
        }
      }
      
      deleteFolderRecursive(nodeCacheDir);
      console.log('✅ Cleaned Node.js cache directory');
    } catch (error) {
      console.error('❌ Error cleaning Node.js cache:', error.message);
    }
  } else {
    console.log('Node.js cache directory not found, skipping');
  }
  
  // Clean dist directory if it exists
  const distDir = path.join(process.cwd(), 'dist');
  if (fs.existsSync(distDir)) {
    try {
      function deleteFolderRecursive(dirPath) {
        if (fs.existsSync(dirPath)) {
          fs.readdirSync(dirPath).forEach((file) => {
            const curPath = path.join(dirPath, file);
            if (fs.lstatSync(curPath).isDirectory()) {
              deleteFolderRecursive(curPath);
            } else {
              fs.unlinkSync(curPath);
            }
          });
          
          // Don't delete the dist directory itself, just its contents
          if (dirPath !== distDir) {
            fs.rmdirSync(dirPath);
          }
        }
      }
      
      deleteFolderRecursive(distDir);
      console.log('✅ Cleaned dist directory');
    } catch (error) {
      console.error('❌ Error cleaning dist directory:', error.message);
    }
  } else {
    console.log('Dist directory not found, skipping');
  }
}

// Clean Python caches
function cleanPythonCaches() {
  console.log('\n=== CLEANING PYTHON CACHES ===');
  
  // Find and remove __pycache__ directories
  try {
    const findResult = execSync('find . -type d -name "__pycache__" -not -path "./node_modules/*" 2>/dev/null || true').toString().trim();
    
    if (findResult) {
      const pycacheDirs = findResult.split('\n');
      console.log(`Found ${pycacheDirs.length} Python cache directories`);
      
      for (const dir of pycacheDirs) {
        try {
          execSync(`rm -rf "${dir}"`);
          console.log(`Deleted ${dir}`);
        } catch (e) {
          console.error(`Error deleting ${dir}:`, e.message);
        }
      }
      
      console.log('✅ Cleaned Python cache directories');
    } else {
      console.log('No Python cache directories found');
    }
    
    // Find and remove .pyc files
    const pycFiles = execSync('find . -name "*.pyc" -not -path "./node_modules/*" 2>/dev/null || true').toString().trim();
    
    if (pycFiles) {
      const files = pycFiles.split('\n');
      console.log(`Found ${files.length} .pyc files`);
      
      for (const file of files) {
        try {
          fs.unlinkSync(file);
        } catch (e) {
          console.error(`Error deleting ${file}:`, e.message);
        }
      }
      
      console.log('✅ Cleaned Python compiled files');
    } else {
      console.log('No Python compiled files found');
    }
  } catch (error) {
    console.error('❌ Error cleaning Python caches:', error.message);
  }
}

// Clean application-specific caches
function cleanApplicationCaches() {
  console.log('\n=== CLEANING APPLICATION CACHES ===');
  
  // Clean Python service cache if present
  try {
    const pythonServicePath = path.join(process.cwd(), 'server', 'python-service.ts');
    
    if (fs.existsSync(pythonServicePath)) {
      const content = fs.readFileSync(pythonServicePath, 'utf8');
      
      if (content.includes('clearCache') || content.includes('clear_cache')) {
        console.log('Found cache clearing code in Python service');
        console.log('Running application with cache clearing flag...');
        
        try {
          // Execute Node script to clear caches in a controlled environment
          const script = `
          const { pythonService } = require('./server/python-service');
          if (pythonService && typeof pythonService.clearCache === 'function') {
            console.log('Clearing Python service cache...');
            pythonService.clearCache();
            console.log('Cache cleared successfully');
          } else {
            console.log('Python service cache clearing function not available');
          }
          process.exit(0);
          `;
          
          const tempScriptPath = path.join(os.tmpdir(), 'clear-python-cache.js');
          fs.writeFileSync(tempScriptPath, script);
          
          execSync(`node ${tempScriptPath}`, { stdio: 'inherit' });
          fs.unlinkSync(tempScriptPath);
          
          console.log('✅ Cleared Python service cache');
        } catch (e) {
          console.error('❌ Error clearing Python service cache:', e.message);
        }
      } else {
        console.log('Python service does not have cache clearing capability');
      }
    } else {
      console.log('Python service file not found, skipping');
    }
  } catch (error) {
    console.error('❌ Error cleaning application caches:', error.message);
  }
}

// Run all cleaning tasks
function runCacheCleaning() {
  console.log('=== CACHE CLEANING STARTING ===');
  console.log(`Date: ${new Date().toISOString()}`);
  console.log(`Platform: ${platform}`);
  console.log(`Node Version: ${process.version}`);
  console.log('-----------------------------------');
  
  cleanTempFiles();
  cleanNodeCaches();
  cleanPythonCaches();
  cleanApplicationCaches();
  
  console.log('\n=== CACHE CLEANING COMPLETE ===');
  console.log('✅ All caches have been cleaned');
  console.log('✅ System should now have improved performance');
}

// Run the cleaning tasks
runCacheCleaning();