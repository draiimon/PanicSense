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

// Cache directories to clean
const cacheDirs = [
  path.join(os.tmpdir(), 'disaster-sentiment'),
  // Linux RAM-based tmpfs
  ...(process.platform === 'linux' && fs.existsSync('/dev/shm') ? 
      [path.join('/dev/shm', 'disaster-sentiment')] : [])
];

console.log('🧹 Starting cache cleanup process...');

// Clean temporary files
cacheDirs.forEach(dir => {
  if (fs.existsSync(dir)) {
    try {
      console.log(`Cleaning directory: ${dir}`);
      const files = fs.readdirSync(dir);
      
      // Get current date for age comparison
      const now = new Date();
      let deletedCount = 0;
      let totalSize = 0;
      
      files.forEach(file => {
        const filePath = path.join(dir, file);
        try {
          const stats = fs.statSync(filePath);
          
          // Delete files older than 24 hours
          const fileAge = (now - stats.mtime) / (1000 * 60 * 60); // age in hours
          if (fileAge > 24) {
            totalSize += stats.size;
            fs.unlinkSync(filePath);
            deletedCount++;
          }
        } catch (e) {
          console.warn(`Could not process file ${filePath}: ${e.message}`);
        }
      });
      
      // Convert bytes to MB for better readability
      const sizeInMB = (totalSize / (1024 * 1024)).toFixed(2);
      console.log(`✅ Deleted ${deletedCount} files (${sizeInMB} MB) from ${dir}`);
    } catch (e) {
      console.error(`Error cleaning directory ${dir}: ${e.message}`);
    }
  } else {
    console.log(`Directory doesn't exist: ${dir}`);
  }
});

// Optional: Clean npm cache if in development environment
if (process.env.NODE_ENV !== 'production') {
  try {
    console.log('Cleaning npm cache...');
    execSync('npm cache clean --force', { stdio: 'inherit' });
    console.log('✅ npm cache cleaned');
  } catch (e) {
    console.warn(`Could not clean npm cache: ${e.message}`);
  }
}

console.log('🎉 Cache cleanup completed successfully!');