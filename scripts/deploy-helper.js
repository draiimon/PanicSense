/**
 * Deployment Helper for PanicSense
 * 
 * This script helps with deployment preparation for both Replit and Render platforms.
 * It ensures consistent environment setup and validates required configurations.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync } = require('child_process');
const { Pool } = require('pg');

const isReplit = Boolean(process.env.REPL_ID);
const isRender = Boolean(process.env.RENDER);
const platform = isRender ? 'Render' : isReplit ? 'Replit' : 'Local';

console.log(`Running deployment helper on ${platform} platform`);

// Validate database connection and schema
async function validateDatabase() {
  if (!process.env.DATABASE_URL) {
    console.error('❌ DATABASE_URL environment variable is missing');
    process.exit(1);
  }

  console.log('Checking database connection...');
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: isRender ? { rejectUnauthorized: false } : false,
    connectionTimeoutMillis: 10000,
  });

  try {
    // Check connection
    const client = await pool.connect();
    console.log('✅ Database connection successful');
    
    // Check tables
    const tables = await client.query(`
      SELECT tablename FROM pg_catalog.pg_tables 
      WHERE schemaname='public'
    `);
    
    console.log(`Found ${tables.rows.length} tables in database`);
    
    // Check required tables
    const requiredTables = [
      'users', 'sessions', 'sentiment_posts', 
      'disaster_events', 'analyzed_files'
    ];
    
    const missingTables = requiredTables.filter(
      table => !tables.rows.some(row => row.tablename === table)
    );
    
    if (missingTables.length > 0) {
      console.log(`⚠️ Missing tables: ${missingTables.join(', ')}`);
      
      // Create missing tables using migration script
      console.log('Attempting to create missing tables...');
      try {
        execSync('npm run db:push');
        console.log('✅ Database schema updated successfully');
      } catch (error) {
        console.error('❌ Failed to create missing tables:', error.message);
      }
    } else {
      console.log('✅ All required tables exist');
    }
    
    client.release();
  } catch (error) {
    console.error('❌ Database connection failed:', error.message);
    console.log('Will continue deployment, but database issues must be resolved');
  } finally {
    await pool.end();
  }
}

// Validate Python setup
function validatePythonSetup() {
  console.log('Checking Python setup...');
  try {
    const pythonBinary = process.env.PYTHON_BINARY || 'python3';
    const pythonVersion = execSync(`${pythonBinary} --version`).toString().trim();
    console.log(`✅ Python detected: ${pythonVersion}`);
    
    // Check if required Python packages are installed
    const requiredPackages = [
      'langdetect', 'nltk', 'numpy', 'pandas', 
      'scikit-learn', 'torch'
    ];
    
    // Check each package with pip list
    const pipList = execSync(`${pythonBinary} -m pip list`).toString();
    const missingPackages = requiredPackages.filter(
      pkg => !pipList.includes(pkg)
    );
    
    if (missingPackages.length > 0) {
      console.log(`⚠️ Missing Python packages: ${missingPackages.join(', ')}`);
      // Attempt to install missing packages
      console.log('Installing missing Python packages...');
      execSync(`${pythonBinary} -m pip install ${missingPackages.join(' ')}`);
      console.log('✅ Python packages installed successfully');
    } else {
      console.log('✅ All required Python packages are installed');
    }
  } catch (error) {
    console.error('⚠️ Python validation issue:', error.message);
    console.log('Will continue deployment, but Python issues should be addressed');
  }
}

// Prepare build directory
function prepareBuildDirectory() {
  console.log('Preparing build directory...');
  const buildDir = path.join(process.cwd(), 'dist');
  
  // Create build dir if it doesn't exist
  if (!fs.existsSync(buildDir)) {
    fs.mkdirSync(buildDir, { recursive: true });
  }
  
  // Ensure assets are copied to the build directory
  const assetsSrc = path.join(process.cwd(), 'attached_assets');
  const assetsDest = path.join(buildDir, 'assets');
  
  if (fs.existsSync(assetsSrc) && !fs.existsSync(assetsDest)) {
    fs.mkdirSync(assetsDest, { recursive: true });
    
    try {
      const files = fs.readdirSync(assetsSrc);
      for (const file of files) {
        const sourcePath = path.join(assetsSrc, file);
        const targetPath = path.join(assetsDest, file);
        fs.copyFileSync(sourcePath, targetPath);
      }
      console.log(`✅ Copied ${files.length} assets to build directory`);
    } catch (error) {
      console.error('⚠️ Asset copying issue:', error.message);
    }
  }
  
  console.log('✅ Build directory prepared');
}

// Prepare temp directories
function prepareTempDirectories() {
  console.log('Preparing temporary directories...');
  
  // Create temp directory for uploads
  const tempDir = path.join(
    isRender || isReplit ? '/tmp' : os.tmpdir(), 
    'disaster-sentiment'
  );
  
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
    console.log(`✅ Created temp directory: ${tempDir}`);
  }
  
  console.log('✅ Temporary directories prepared');
}

// Render-specific setup
function prepareForRender() {
  console.log('Preparing for Render deployment...');
  
  // Set production environment
  process.env.NODE_ENV = 'production';
  
  // Ensure PORT is set correctly
  if (!process.env.PORT) {
    process.env.PORT = '5000';
  }
  
  console.log('✅ Render preparation complete');
}

// Replit-specific setup
function prepareForReplit() {
  console.log('Preparing for Replit deployment...');
  
  // Set production environment for deployment
  process.env.NODE_ENV = 'production';
  
  console.log('✅ Replit preparation complete');
}

async function runDeploymentChecks() {
  console.log('=== Deployment Helper Starting ===');
  
  // Run platform-specific setup
  if (isRender) {
    prepareForRender();
  } else if (isReplit) {
    prepareForReplit();
  }
  
  // Run common validations
  await validateDatabase();
  validatePythonSetup();
  prepareBuildDirectory();
  prepareTempDirectories();
  
  console.log('=== Deployment Preparation Complete ===');
  console.log('You can now deploy the application to the target platform');
}

// Run the deployment helper
runDeploymentChecks().catch(error => {
  console.error('Deployment helper failed:', error);
  process.exit(1);
});