#!/usr/bin/env node

/**
 * Cross-Platform Compatibility Helper for PanicSense
 * Author: Mark Andrei R. Castillo
 * 
 * This script ensures the application runs correctly across local development,
 * Replit, and Render environments by checking for common issues and fixing them.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const PLATFORM = process.env.PLATFORM || (process.env.REPL_ID ? 'replit' : (process.env.RENDER ? 'render' : 'local'));
console.log(`Detected platform: ${PLATFORM}`);

// Ensure required directories exist
const requiredDirs = [
  'assets',
  'assets/icons',
  'assets/screenshots',
  'client/public',
  'client/src',
  'server/python'
];

requiredDirs.forEach(dir => {
  const dirPath = path.join(process.cwd(), dir);
  if (!fs.existsSync(dirPath)) {
    console.log(`Creating directory: ${dir}`);
    fs.mkdirSync(dirPath, { recursive: true });
  }
});

// Create empty .env file if it doesn't exist
const envPath = path.join(process.cwd(), '.env');
if (!fs.existsSync(envPath)) {
  console.log('Creating empty .env file from example...');
  try {
    const envExamplePath = path.join(process.cwd(), '.env.example');
    if (fs.existsSync(envExamplePath)) {
      fs.copyFileSync(envExamplePath, envPath);
    } else {
      // Create minimal .env file
      fs.writeFileSync(envPath, 
        'PORT=5000\n' +
        'DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres\n' +
        'GROQ_API_KEY_1=your_groq_api_key_here\n' +
        'VALIDATION_API_KEY=your_validation_api_key_here\n'
      );
    }
  } catch (error) {
    console.error('Error creating .env file:', error);
  }
}

// Platform-specific adjustments
if (PLATFORM === 'replit') {
  console.log('Applying Replit-specific configurations...');
  
  // Replit-specific configuration
  const replitConfig = {
    run: "npm start",
    compile: "npm run build",
    packager: "npm"
  };
  
  try {
    // Check if .replit exists and needs modification
    const replitConfigPath = path.join(process.cwd(), '.replit');
    if (!fs.existsSync(replitConfigPath)) {
      fs.writeFileSync(replitConfigPath, Object.entries(replitConfig)
        .map(([key, value]) => `${key} = "${value}"`)
        .join('\n')
      );
    }
  } catch (error) {
    console.error('Error configuring Replit files:', error);
  }
} else if (PLATFORM === 'render') {
  console.log('Applying Render-specific configurations...');
  // Render uses the Docker configuration automatically
} else {
  console.log('Applying local development configurations...');
}

// Ensure database schema completeness
console.log('Verifying database schema compatibility...');
try {
  const schemaPath = path.join(process.cwd(), 'migrations', 'complete_schema.sql');
  if (!fs.existsSync(schemaPath)) {
    console.error('Error: complete_schema.sql is missing. This file is required for cross-platform compatibility.');
    process.exit(1);
  }
} catch (error) {
  console.error('Error checking schema files:', error);
}

// Check if Python requirements are installed
console.log('Ensuring Python dependencies are available...');
try {
  const pythonRequirementsPath = path.join(process.cwd(), 'server', 'python', 'requirements.txt');
  if (fs.existsSync(pythonRequirementsPath)) {
    console.log('Python requirements file found.');
  } else {
    console.warn('Warning: Python requirements file not found at server/python/requirements.txt');
  }
} catch (error) {
  console.error('Error checking Python requirements:', error);
}

console.log('Cross-platform compatibility check complete!');