/**
 * Render.com Setup Helper
 * This script helps validate and setup PanicSense in the Render.com environment
 */

import fs from 'fs';
import { execSync } from 'child_process';
import pg from 'pg';
import dotenv from 'dotenv';

const { Pool } = pg;

// Load environment variables
dotenv.config();

console.log('🚀 Running PanicSense Render.com setup script...');

// Validate environment
const requiredVars = ['DATABASE_URL', 'PORT'];
const missingVars = requiredVars.filter(varName => !process.env[varName]);

if (missingVars.length > 0) {
  console.error(`❌ Missing required environment variables: ${missingVars.join(', ')}`);
  console.error('Please set these variables in your Render.com dashboard.');
  process.exit(1);
}

// Setup SSL for database
let databaseUrl = process.env.DATABASE_URL;

// Check if the DATABASE_URL already has SSL parameters
if (!databaseUrl.includes('sslmode=')) {
  databaseUrl = `${databaseUrl}?sslmode=require`;
  console.log('✅ Added SSL mode to database URL');
}

// Test database connection
async function testDbConnection() {
  console.log('🔄 Testing database connection...');
  try {
    const pool = new Pool({ 
      connectionString: databaseUrl,
      ssl: process.env.DB_SSL_REQUIRED === 'true' ? { rejectUnauthorized: false } : false
    });
    
    const client = await pool.connect();
    const result = await client.query('SELECT NOW()');
    console.log(`✅ Successfully connected to database at ${result.rows[0].now}`);
    client.release();
    await pool.end();
    return true;
  } catch (error) {
    console.error('❌ Failed to connect to database:', error.message);
    console.error('Please check your DATABASE_URL and make sure your database is accessible.');
    return false;
  }
}

// Create necessary directories
function setupDirectories() {
  const directories = ['logs', 'uploads'];
  directories.forEach(dir => {
    if (!fs.existsSync(dir)) {
      console.log(`📁 Creating directory: ${dir}`);
      fs.mkdirSync(dir, { recursive: true });
    }
  });
}

// Setup static files
function setupStaticFiles() {
  // Verify the build was successful
  if (!fs.existsSync('./dist')) {
    console.error('❌ Build artifacts not found. Something went wrong during the build step.');
    return false;
  }

  // Setup server/public if it doesn't exist
  if (!fs.existsSync('./server/public')) {
    console.log('📁 Creating server/public directory');
    fs.mkdirSync('./server/public', { recursive: true });
    
    // Copy client files if they exist
    if (fs.existsSync('./dist/public')) {
      console.log('📋 Copying dist/public to server/public');
      execSync('cp -r ./dist/public/* ./server/public/');
    } else if (fs.existsSync('./client/dist')) {
      console.log('📋 Copying client/dist to server/public');
      execSync('cp -r ./client/dist/* ./server/public/');
    }
  }
  
  return true;
}

// Run all setup steps
async function runSetup() {
  setupDirectories();
  
  const staticFilesOk = setupStaticFiles();
  if (!staticFilesOk) {
    console.warn('⚠️ There were issues with static files, but continuing anyway...');
  }
  
  const dbOk = await testDbConnection();
  if (!dbOk) {
    console.error('❌ Database connection failed. Application may not work properly.');
    console.error('Please check your DATABASE_URL environment variable.');
  }
  
  console.log('✅ Render.com setup completed!');
  console.log('🚀 Ready to start the application');
}

runSetup().catch(err => {
  console.error('❌ Fatal error during setup:', err);
  process.exit(1);
});