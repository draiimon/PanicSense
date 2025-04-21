/**
 * Render.com Setup Helper
 * This script helps validate and setup PanicSense in the Render.com environment
 */

import pg from 'pg';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const { Pool } = pg;

// Test database connection to make sure it's working
async function testDbConnection() {
  console.log('🔄 Testing database connection...');
  
  const databaseUrl = process.env.DATABASE_URL || '';
  if (!databaseUrl) {
    console.error('❌ No DATABASE_URL found in environment variables');
    return false;
  }
  
  try {
    const pool = new Pool({
      connectionString: databaseUrl,
      ssl: { rejectUnauthorized: false }
    });
    
    const client = await pool.connect();
    const result = await client.query('SELECT NOW()');
    console.log(`✅ Database connection successful: ${result.rows[0].now}`);
    client.release();
    await pool.end();
    return true;
  } catch (error) {
    console.error('❌ Database connection failed:', error.message);
    return false;
  }
}

// Setup directories for static files and uploads
function setupDirectories() {
  const dirs = [
    path.join(__dirname, 'uploads'),
    path.join(__dirname, 'logs'),
    path.join(__dirname, 'server', 'public')
  ];
  
  dirs.forEach(dir => {
    if (!fs.existsSync(dir)) {
      try {
        fs.mkdirSync(dir, { recursive: true });
        console.log(`✅ Created directory: ${dir}`);
      } catch (error) {
        console.error(`❌ Failed to create directory ${dir}:`, error.message);
      }
    } else {
      console.log(`✅ Directory already exists: ${dir}`);
    }
  });
}

// Make sure static files are in the right place
function setupStaticFiles() {
  // Only needed if client-side files aren't already in the right place
  const clientDist = path.join(__dirname, 'client', 'dist');
  const serverPublic = path.join(__dirname, 'server', 'public');
  const distPublic = path.join(__dirname, 'dist', 'public');
  
  if (fs.existsSync(clientDist)) {
    try {
      console.log('📦 Copying client/dist to server/public...');
      // This is a simplified version - in a real implementation you'd need a recursive copy function
      fs.readdirSync(clientDist).forEach(file => {
        const srcPath = path.join(clientDist, file);
        const destPath = path.join(serverPublic, file);
        
        if (fs.statSync(srcPath).isFile()) {
          fs.copyFileSync(srcPath, destPath);
        }
      });
      console.log('✅ Client files copied successfully');
    } catch (error) {
      console.error('❌ Failed to copy client files:', error.message);
    }
  } else if (fs.existsSync(distPublic)) {
    try {
      console.log('📦 Copying dist/public to server/public...');
      fs.readdirSync(distPublic).forEach(file => {
        const srcPath = path.join(distPublic, file);
        const destPath = path.join(serverPublic, file);
        
        if (fs.statSync(srcPath).isFile()) {
          fs.copyFileSync(srcPath, destPath);
        }
      });
      console.log('✅ Dist files copied successfully');
    } catch (error) {
      console.error('❌ Failed to copy dist files:', error.message);
    }
  } else {
    console.log('ℹ️ No client or dist files found, skipping copy');
  }
}

// Main setup function
async function runSetup() {
  console.log('======================================');
  console.log('🚀 Starting Render.com setup process');
  console.log('======================================');
  
  // Run each setup step
  setupDirectories();
  setupStaticFiles();
  await testDbConnection();
  
  console.log('======================================');
  console.log('✅ Render.com setup process complete');
  console.log('======================================');
}

// Run setup and catch any unhandled errors
runSetup().catch(error => {
  console.error('❌ Unhandled error in render-setup.js:', error);
  process.exit(1);
});