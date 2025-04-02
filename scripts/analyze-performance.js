/**
 * Performance Analysis Script for PanicSense
 * 
 * This script analyzes system and application performance,
 * identifies bottlenecks, and provides optimization recommendations.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync } = require('child_process');

const isReplit = Boolean(process.env.REPL_ID);
const isRender = Boolean(process.env.RENDER);
const platform = isRender ? 'Render' : isReplit ? 'Replit' : 'Local';

console.log(`Running performance analysis on ${platform} platform`);

// Check database performance
function checkDatabasePerformance() {
  console.log('\n=== DATABASE PERFORMANCE ANALYSIS ===');
  
  if (!process.env.DATABASE_URL) {
    console.log('❌ DATABASE_URL not found, skipping database analysis');
    return {
      status: 'error',
      recommendations: [
        'Set DATABASE_URL environment variable'
      ]
    };
  }
  
  try {
    // Check connection pool settings
    const dbModule = path.join(process.cwd(), 'server', 'db.ts');
    if (fs.existsSync(dbModule)) {
      const dbContent = fs.readFileSync(dbModule, 'utf8');
      
      // Check for pool configuration
      const hasPoolConfig = dbContent.includes('new Pool({');
      const hasMaxConnections = dbContent.includes('max:');
      const hasIdleTimeout = dbContent.includes('idleTimeoutMillis:');
      const hasConnectionTimeout = dbContent.includes('connectionTimeoutMillis:');
      
      console.log('Database Connection Pool Settings:');
      console.log(`- Custom pool configuration: ${hasPoolConfig ? '✅' : '❌'}`);
      console.log(`- Max connections limit: ${hasMaxConnections ? '✅' : '❌'}`);
      console.log(`- Idle timeout: ${hasIdleTimeout ? '✅' : '❌'}`);
      console.log(`- Connection timeout: ${hasConnectionTimeout ? '✅' : '❌'}`);
      
      const recommendations = [];
      if (!hasPoolConfig) {
        recommendations.push('Implement custom pool configuration for better performance');
      }
      if (!hasMaxConnections) {
        recommendations.push('Set max connections based on available system resources');
      }
      if (!hasIdleTimeout) {
        recommendations.push('Set idleTimeoutMillis to release unused connections');
      }
      if (!hasConnectionTimeout) {
        recommendations.push('Set connectionTimeoutMillis to avoid hanging connections');
      }
      
      return {
        status: recommendations.length > 0 ? 'warning' : 'good',
        recommendations
      };
    } else {
      console.log('❌ Database module not found');
      return {
        status: 'error',
        recommendations: [
          'Create a proper database module with optimized connection pool'
        ]
      };
    }
  } catch (error) {
    console.error('Error analyzing database performance:', error.message);
    return {
      status: 'error',
      recommendations: [
        'Fix database configuration errors',
        'Ensure database module is properly implemented'
      ]
    };
  }
}

// Check file system performance
function checkFileSystem() {
  console.log('\n=== FILE SYSTEM ANALYSIS ===');
  
  try {
    console.log('File System Information:');
    console.log(`- Platform: ${os.platform()}`);
    console.log(`- Architecture: ${os.arch()}`);
    console.log(`- Total Memory: ${Math.round(os.totalmem() / (1024 * 1024))} MB`);
    console.log(`- Free Memory: ${Math.round(os.freemem() / (1024 * 1024))} MB`);
    
    // Check temp directory setup
    const tempDir = process.env.TEMP_DIR || 
                   (isRender || isReplit ? '/tmp/disaster-sentiment' : 
                   path.join(os.tmpdir(), 'disaster-sentiment'));
    
    const tempDirExists = fs.existsSync(tempDir);
    console.log(`- Temp directory (${tempDir}): ${tempDirExists ? '✅' : '❌'}`);
    
    // Check assets directory
    const assetsDir = path.join(process.cwd(), 'assets');
    const attachedAssetsDir = path.join(process.cwd(), 'attached_assets');
    
    const assetsDirExists = fs.existsSync(assetsDir);
    const attachedAssetsDirExists = fs.existsSync(attachedAssetsDir);
    
    console.log(`- Assets directory: ${assetsDirExists ? '✅' : '❌'}`);
    console.log(`- Attached assets directory: ${attachedAssetsDirExists ? '✅' : '❌'}`);
    
    // Determine if symbolic link is used
    let assetsIsSymlink = false;
    if (assetsDirExists) {
      try {
        const stats = fs.lstatSync(assetsDir);
        assetsIsSymlink = stats.isSymbolicLink();
        console.log(`- Assets is symbolic link: ${assetsIsSymlink ? '✅' : '❌'}`);
      } catch (e) {
        console.log('- Could not determine if assets is a symbolic link');
      }
    }
    
    const recommendations = [];
    if (!tempDirExists) {
      recommendations.push('Create temp directory for file uploads and processing');
    }
    
    if (!assetsDirExists && attachedAssetsDirExists) {
      recommendations.push('Create assets directory as symbolic link to attached_assets for better performance');
    }
    
    if (assetsDirExists && !assetsIsSymlink && attachedAssetsDirExists) {
      recommendations.push('Convert assets directory to symbolic link to attached_assets to save disk space');
    }
    
    return {
      status: recommendations.length > 0 ? 'warning' : 'good',
      recommendations
    };
  } catch (error) {
    console.error('Error analyzing file system:', error.message);
    return {
      status: 'error',
      recommendations: [
        'Fix file system configuration'
      ]
    };
  }
}

// Check Node.js optimizations
function checkNodeOptimizations() {
  console.log('\n=== NODE.JS OPTIMIZATIONS ===');
  
  try {
    console.log('Node.js Environment:');
    console.log(`- Node Version: ${process.version}`);
    console.log(`- NODE_ENV: ${process.env.NODE_ENV || 'not set'}`);
    
    // Check compression middleware
    const serverFile = path.join(process.cwd(), 'server', 'index.ts');
    let hasCompression = false;
    let hasCors = false;
    let hasErrorHandling = false;
    
    if (fs.existsSync(serverFile)) {
      const serverContent = fs.readFileSync(serverFile, 'utf8');
      hasCompression = serverContent.includes('compression');
      hasCors = serverContent.includes('cors');
      hasErrorHandling = serverContent.includes('app.use((err');
      
      console.log(`- Compression middleware: ${hasCompression ? '✅' : '❌'}`);
      console.log(`- CORS middleware: ${hasCors ? '✅' : '❌'}`);
      console.log(`- Error handling middleware: ${hasErrorHandling ? '✅' : '❌'}`);
    } else {
      console.log('❌ Server index file not found');
    }
    
    // Check for platform adapter
    const platformAdapterFile = path.join(process.cwd(), 'server', 'utils', 'platform-adapter.ts');
    const hasPlatformAdapter = fs.existsSync(platformAdapterFile);
    console.log(`- Platform adapter: ${hasPlatformAdapter ? '✅' : '❌'}`);
    
    // Check package.json for optimization scripts
    const packageJsonPath = path.join(process.cwd(), 'package.json');
    let hasOptimizationScripts = false;
    
    if (fs.existsSync(packageJsonPath)) {
      const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
      const scripts = packageJson.scripts || {};
      
      hasOptimizationScripts = 
        Object.keys(scripts).some(key => 
          key.includes('build') || 
          key.includes('optimize') ||
          key.includes('db:push'));
      
      console.log(`- Build/optimization scripts: ${hasOptimizationScripts ? '✅' : '❌'}`);
    }
    
    const recommendations = [];
    if (!hasCompression) {
      recommendations.push('Add compression middleware for faster response times');
    }
    if (!hasCors) {
      recommendations.push('Add CORS middleware for API access');
    }
    if (!hasErrorHandling) {
      recommendations.push('Implement proper error handling middleware');
    }
    if (!hasPlatformAdapter) {
      recommendations.push('Implement platform adapter for consistent behavior across environments');
    }
    if (!hasOptimizationScripts) {
      recommendations.push('Add build and optimization scripts to package.json');
    }
    
    return {
      status: recommendations.length > 0 ? 'warning' : 'good',
      recommendations
    };
  } catch (error) {
    console.error('Error analyzing Node.js optimizations:', error.message);
    return {
      status: 'error',
      recommendations: [
        'Fix Node.js configuration issues'
      ]
    };
  }
}

// Check Python optimizations
function checkPythonOptimizations() {
  console.log('\n=== PYTHON OPTIMIZATIONS ===');
  
  try {
    // Check if Python is available
    let pythonVersion = 'not detected';
    try {
      pythonVersion = execSync('python3 --version 2>/dev/null || python --version 2>/dev/null').toString().trim();
    } catch (e) {
      console.log('❌ Python not installed or not found in PATH');
    }
    
    console.log(`- Python version: ${pythonVersion}`);
    
    // Check Python service implementation
    const pythonServiceFile = path.join(process.cwd(), 'server', 'python-service.ts');
    let hasPythonService = false;
    let hasCaching = false;
    let hasQueue = false;
    
    if (fs.existsSync(pythonServiceFile)) {
      const pythonServiceContent = fs.readFileSync(pythonServiceFile, 'utf8');
      hasPythonService = true;
      hasCaching = pythonServiceContent.includes('cache') || 
                   pythonServiceContent.includes('Cache');
      hasQueue = pythonServiceContent.includes('queue') || 
                 pythonServiceContent.includes('Queue');
      
      console.log(`- Python service: ✅`);
      console.log(`- Result caching: ${hasCaching ? '✅' : '❌'}`);
      console.log(`- Processing queue: ${hasQueue ? '✅' : '❌'}`);
    } else {
      console.log('❌ Python service not found');
    }
    
    const recommendations = [];
    if (!hasPythonService) {
      recommendations.push('Implement Python service for AI processing');
    }
    if (!hasCaching) {
      recommendations.push('Add caching to Python service for better performance');
    }
    if (!hasQueue) {
      recommendations.push('Implement processing queue in Python service to prevent overload');
    }
    
    return {
      status: recommendations.length > 0 ? 'warning' : 'good',
      recommendations
    };
  } catch (error) {
    console.error('Error analyzing Python optimizations:', error.message);
    return {
      status: 'error',
      recommendations: [
        'Fix Python configuration issues'
      ]
    };
  }
}

// Check application-specific optimizations
function checkApplicationOptimizations() {
  console.log('\n=== APPLICATION OPTIMIZATIONS ===');
  
  try {
    // Check for WebSocket implementation
    const routesFile = path.join(process.cwd(), 'server', 'routes.ts');
    let hasWebSockets = false;
    let hasProgressTracking = false;
    let hasUploadHandling = false;
    
    if (fs.existsSync(routesFile)) {
      const routesContent = fs.readFileSync(routesFile, 'utf8');
      hasWebSockets = routesContent.includes('WebSocket');
      hasProgressTracking = routesContent.includes('progress') || 
                            routesContent.includes('Progress');
      hasUploadHandling = routesContent.includes('upload') || 
                          routesContent.includes('Upload') ||
                          routesContent.includes('multer');
      
      console.log(`- WebSocket implementation: ${hasWebSockets ? '✅' : '❌'}`);
      console.log(`- Progress tracking: ${hasProgressTracking ? '✅' : '❌'}`);
      console.log(`- File upload handling: ${hasUploadHandling ? '✅' : '❌'}`);
    } else {
      console.log('❌ Routes file not found');
    }
    
    // Check clear-cache script
    const clearCacheScript = path.join(process.cwd(), 'scripts', 'clear-cache.js');
    const hasClearCacheScript = fs.existsSync(clearCacheScript);
    console.log(`- Cache cleaning script: ${hasClearCacheScript ? '✅' : '❌'}`);
    
    // Check health check endpoint
    let hasHealthCheck = false;
    if (fs.existsSync(routesFile)) {
      const routesContent = fs.readFileSync(routesFile, 'utf8');
      hasHealthCheck = routesContent.includes('/api/health');
      console.log(`- Health check endpoint: ${hasHealthCheck ? '✅' : '❌'}`);
    }
    
    const recommendations = [];
    if (!hasWebSockets) {
      recommendations.push('Implement WebSockets for real-time updates');
    }
    if (!hasProgressTracking) {
      recommendations.push('Add progress tracking for long-running operations');
    }
    if (!hasUploadHandling) {
      recommendations.push('Implement proper file upload handling with progress reporting');
    }
    if (!hasClearCacheScript) {
      recommendations.push('Create cache cleaning script for maintenance');
    }
    if (!hasHealthCheck) {
      recommendations.push('Add health check endpoint for monitoring');
    }
    
    return {
      status: recommendations.length > 0 ? 'warning' : 'good',
      recommendations
    };
  } catch (error) {
    console.error('Error analyzing application optimizations:', error.message);
    return {
      status: 'error',
      recommendations: [
        'Fix application configuration issues'
      ]
    };
  }
}

// Run performance analysis
function runPerformanceAnalysis() {
  console.log('=== PERFORMANCE ANALYSIS STARTING ===');
  console.log(`Date: ${new Date().toISOString()}`);
  console.log(`Platform: ${platform}`);
  console.log(`Node Version: ${process.version}`);
  console.log(`Architecture: ${os.arch()}`);
  console.log('-----------------------------------');
  
  const dbPerformance = checkDatabasePerformance();
  const fileSystemPerformance = checkFileSystem();
  const nodeOptimizations = checkNodeOptimizations();
  const pythonOptimizations = checkPythonOptimizations();
  const appOptimizations = checkApplicationOptimizations();
  
  console.log('\n=== PERFORMANCE ANALYSIS SUMMARY ===');
  console.log(`Database Performance: ${dbPerformance.status}`);
  console.log(`File System Optimization: ${fileSystemPerformance.status}`);
  console.log(`Node.js Optimizations: ${nodeOptimizations.status}`);
  console.log(`Python Optimizations: ${pythonOptimizations.status}`);
  console.log(`Application Optimizations: ${appOptimizations.status}`);
  
  console.log('\n=== RECOMMENDATIONS ===');
  
  let allRecommendations = [
    ...dbPerformance.recommendations,
    ...fileSystemPerformance.recommendations,
    ...nodeOptimizations.recommendations,
    ...pythonOptimizations.recommendations,
    ...appOptimizations.recommendations
  ];
  
  if (allRecommendations.length === 0) {
    console.log('✅ No performance issues detected. Application is well optimized!');
  } else {
    allRecommendations.forEach((rec, index) => {
      console.log(`${index + 1}. ${rec}`);
    });
  }
  
  console.log('\n=== PERFORMANCE ANALYSIS COMPLETE ===');
}

// Run the analysis
runPerformanceAnalysis();