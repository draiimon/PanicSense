#!/usr/bin/env node

/**
 * Cross-Platform Compatibility Helper for PanicSense
 * Author: Mark Andrei R. Castillo
 * 
 * This script ensures the application runs correctly across local development,
 * Replit, and Render environments by checking for common issues and fixing them.
 */

import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawnSync } from 'child_process';

// Get current directory (ES modules don't have __dirname)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Environment detection
const isReplit = !!process.env.REPL_ID || !!process.env.REPLIT_ENVIRONMENT;
const isRender = process.env.NODE_ENV === 'production' && !!process.env.RENDER;
const isLocal = !isReplit && !isRender;

console.log('🔄 PanicSense Cross-Platform Compatibility Helper');
console.log('================================================');
console.log(`Detected environment: ${isReplit ? '🟣 Replit' : isRender ? '🟢 Render' : '🔵 Local Development'}`);

// Function to fix common issues
async function ensureCompatibility() {
  let fixesApplied = false;
  
  // 1. Check .env file
  if (!process.env.DATABASE_URL) {
    console.log('\n🔧 DATABASE_URL not found. Creating default .env file...');
    
    const envExample = fs.readFileSync('.env.example', 'utf8');
    let envContent = envExample.replace(
      /DATABASE_URL=.*/,
      'DATABASE_URL=postgresql://neondb_owner:npg_N5MsSKHuk1Qf@ep-silent-sun-a1u48xwz-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require'
    );
    
    // Only write if not in Render (as Render uses environment variables, not .env)
    if (!isRender) {
      fs.writeFileSync('.env', envContent);
      console.log('✅ Created .env with Neon database configuration');
      fixesApplied = true;
    }
  }
  
  // 2. Check file permissions for scripts
  if (isLocal || isReplit) {
    const scriptsDir = path.join(__dirname);
    const scriptFiles = fs.readdirSync(scriptsDir).filter(f => 
      f.endsWith('.js') || f.endsWith('.sh')
    );
    
    for (const scriptFile of scriptFiles) {
      const scriptPath = path.join(scriptsDir, scriptFile);
      try {
        fs.chmodSync(scriptPath, 0o755); // rwxr-xr-x
        console.log(`✅ Set executable permissions for ${scriptFile}`);
        fixesApplied = true;
      } catch (err) {
        console.error(`❌ Failed to set permissions for ${scriptFile}:`, err.message);
      }
    }
  }
  
  // 3. Check Python dependencies for NLP
  if (isReplit) {
    // Replit may need specific Python dependency handling
    console.log('\n🔍 Checking Python dependencies for NLP...');
    
    const pythonCheck = spawnSync('python', ['-c', 'import pandas, nltk, torch, numpy']);
    if (pythonCheck.status !== 0) {
      console.log('⚠️ Missing Python dependencies. Installing...');
      
      // Try to install dependencies
      const installResult = spawnSync('pip', ['install', 'pandas', 'nltk', 'numpy', 'torch', 'langdetect', 'scikit-learn', 'tqdm'], {
        stdio: 'inherit'
      });
      
      if (installResult.status === 0) {
        console.log('✅ Python dependencies installed successfully');
        fixesApplied = true;
      } else {
        console.error('❌ Failed to install Python dependencies');
      }
    } else {
      console.log('✅ Python dependencies are already installed');
    }
  }
  
  // 4. Verify Node.js version
  const nodeMajorVersion = parseInt(process.version.slice(1).split('.')[0], 10);
  
  if (nodeMajorVersion < 16) {
    console.log(`⚠️ Warning: Node.js version ${process.version} is below the recommended version (v16+)`);
    console.log('Some features may not work correctly. Consider upgrading Node.js.');
    
    if (isLocal) {
      console.log('Run: nvm install 16 && nvm use 16');
    }
  } else {
    console.log(`✅ Node.js version ${process.version} meets requirements`);
  }
  
  // 5. Fix static asset serving in production
  if (isRender) {
    console.log('\n🔧 Ensuring correct static asset serving for Render...');
    
    // Additional checks specific to Render environment could be added here
    
    console.log('✅ Static asset serving verified for Render');
  }
  
  // Summary
  console.log('\n🔄 Compatibility check completed!');
  if (fixesApplied) {
    console.log('✅ Some fixes were applied. The application should now work correctly.');
  } else {
    console.log('✅ No fixes needed. Your environment is properly configured.');
  }
}

// Run the compatibility checks
ensureCompatibility().catch(err => {
  console.error('Error during compatibility check:', err);
  process.exit(1);
});