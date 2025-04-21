/**
 * PRE-RENDER START
 * This script runs before the server starts on Render
 * It applies critical fixes to ensure the server can run correctly
 */

import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { emergencyDatabaseFix } from './server/direct-db-fix.js';

// Load environment variables
dotenv.config();

console.log('✅ Preparing server environment...');

// Check if client/dist exists, create if needed
const clientDistPath = path.join(process.cwd(), 'client', 'dist');
if (!fs.existsSync(clientDistPath)) {
  console.log('📂 Creating client/dist directory');
  fs.mkdirSync(clientDistPath, { recursive: true });
}

// Check if uploads directory exists, create if needed
const uploadsPath = path.join(process.cwd(), 'uploads');
if (!fs.existsSync(uploadsPath)) {
  console.log('📂 Creating uploads directory');
  fs.mkdirSync(uploadsPath, { recursive: true });
}

console.log('📂 Using client/dist for static files');

// Disable SSL verification for Python scripts in Render environment
if (process.env.DISABLE_SSL_VERIFY === 'true') {
  console.log('🔒 Disabling SSL verification for Python scripts');
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
}

// Ensure DATABASE_URL has SSL mode if needed
if (process.env.DATABASE_URL && process.env.DB_SSL_REQUIRED === 'true') {
  if (!process.env.DATABASE_URL.includes('sslmode=')) {
    process.env.DATABASE_URL = process.env.DATABASE_URL + '?sslmode=require';
    console.log('🔒 Added sslmode=require to DATABASE_URL');
  } else {
    console.log('🔒 SSL mode already present in DATABASE_URL');
  }
}

// Set safe default environment variables
if (!process.env.NODE_ENV) {
  process.env.NODE_ENV = 'production';
}
console.log(`⚙️ Environment: NODE_ENV=${process.env.NODE_ENV}`);

// Run database fix
console.log('🔄 Attempting database connection...');
emergencyDatabaseFix().then((success) => {
  if (success) {
    console.log('✅ Database fixes applied successfully');
  } else {
    console.log('⚠️ Database connection issues or fixes not applied');
  }
  
  console.log('🚀 Starting server on port 10000...');
  
  // Import and execute server.js
  import('./server.js').catch(error => {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  });
}).catch(error => {
  console.error('❌ Database fix failed:', error);
  
  // Continue to server even if database fix fails
  console.log('🚀 Starting server on port 10000 despite database issues...');
  
  import('./server.js').catch(error => {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  });
});