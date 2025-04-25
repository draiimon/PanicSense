#!/usr/bin/env node

/**
 * Special runner script for Render deployment
 * This is a simplified entry point that ensures the correct files are loaded
 */

// Import required modules
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync } from 'fs';
import { spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('========================================');
console.log(`🚀 PanicSense Starter for Render Deployment`);
console.log(`Starting at: ${new Date().toISOString()}`);
console.log(`Node version: ${process.version}`);
console.log(`Current directory: ${process.cwd()}`);
console.log('========================================');

// First, ensure our database connection works
console.log('Checking database connection...');
if (!process.env.DATABASE_URL) {
  console.warn('⚠️ No DATABASE_URL environment variable found! App may not function correctly.');
}

// Try multiple possible entry points
const possibleEntryPoints = [
  './server/index-wrapper.js',
  './server/index.js',
  './index.js',
  './server.js',
  './dist/index.js'
];

// First, check if all the files exist and log their status
console.log('Checking possible entry points:');
for (const entryPoint of possibleEntryPoints) {
  console.log(`- ${entryPoint}: ${existsSync(join(__dirname, entryPoint)) ? '✅ exists' : '❌ missing'}`);
}

// Try to find the best entry point
let entryPointFound = false;

// Create any missing critical directories
const criticalDirs = ['uploads', 'uploads/temp', 'uploads/data', 'uploads/profile_images', 'python', 'dist/public'];
for (const dir of criticalDirs) {
  const dirPath = join(__dirname, dir);
  if (!existsSync(dirPath)) {
    console.log(`Creating missing directory: ${dir}`);
    try {
      import('fs').then(fs => {
        fs.mkdirSync(dirPath, { recursive: true });
      });
    } catch (err) {
      console.warn(`Failed to create ${dir} directory:`, err.message);
    }
  }
}

for (const entryPoint of possibleEntryPoints) {
  if (existsSync(join(__dirname, entryPoint))) {
    console.log(`✅ Found entry point: ${entryPoint}`);
    console.log(`Starting server from: ${entryPoint}`);
    
    // Use dynamic import to start the server
    import(entryPoint)
      .then(() => {
        console.log(`Server started successfully via ${entryPoint}`);
      })
      .catch(err => {
        console.error(`❌ Failed to start server from ${entryPoint} with ESM import:`, err);
        console.log('Trying to start with child process...');
        
        // If dynamic import fails, try using child process
        try {
          const child = spawn('node', [entryPoint], {
            stdio: 'inherit',
            env: process.env
          });
          
          child.on('error', (err) => {
            console.error(`Child process error for ${entryPoint}:`, err);
          });
          
          child.on('exit', (code) => {
            if (code !== 0) {
              console.error(`Child process for ${entryPoint} exited with code ${code}`);
              createEmergencyServer();
            }
          });
          
          // Don't exit this process
          return;
        } catch (spawnErr) {
          console.error(`Failed to spawn child process for ${entryPoint}:`, spawnErr);
          // Continue to try next entry point
        }
      });
    
    entryPointFound = true;
    break;
  }
}

if (!entryPointFound) {
  console.error('❌ No valid entry point found! Tried:', possibleEntryPoints);
  createEmergencyServer();
}

// Function to create an emergency server if all else fails
function createEmergencyServer() {
  console.error('Creating emergency server...');
  
  // Create an emergency basic server if no entry point is found
  import('express')
    .then(({ default: express }) => {
      const app = express();
      
      app.get('/', (_req, res) => {
        res.send(`
          <html>
            <head>
              <title>PanicSense - Emergency Mode</title>
              <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; padding: 20px; line-height: 1.6; }
                header { background-color: #e74c3c; color: white; padding: 20px; border-radius: 5px; }
                h1 { margin: 0; }
                .content { margin-top: 20px; }
                .api-box { background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #e74c3c; margin: 20px 0; }
                .diagnostics { font-family: monospace; background: #f8f9fa; padding: 15px; overflow-x: auto; }
              </style>
            </head>
            <body>
              <header>
                <h1>PanicSense - Emergency Mode</h1>
              </header>
              <div class="content">
                <p>The PanicSense server is running in emergency mode.</p>
                <div class="api-box">
                  <h3>Limited Functionality</h3>
                  <p>Only basic API endpoints are available.</p>
                  <p>For full functionality, please check the server configuration.</p>
                </div>
                <h3>Server Diagnostics</h3>
                <div class="diagnostics">
                  <p>Node version: ${process.version}</p>
                  <p>Time: ${new Date().toISOString()}</p>
                  <p>Deployment mode: ${process.env.NODE_ENV || 'development'}</p>
                  <p>Port: ${process.env.PORT || '10000'}</p>
                  <p>Checked entry points: ${possibleEntryPoints.join(', ')}</p>
                  <p>Database URL: ${process.env.DATABASE_URL ? '✅ Set' : '❌ Missing'}</p>
                </div>
              </div>
            </body>
          </html>
        `);
      });
      
      app.get('/api/health', (_req, res) => {
        res.json({ 
          status: 'ok', 
          mode: 'emergency',
          timestamp: new Date().toISOString(),
          nodeVersion: process.version,
          environment: process.env.NODE_ENV || 'development',
          databaseConnected: !!process.env.DATABASE_URL
        });
      });
      
      const port = parseInt(process.env.PORT || '10000');
      app.listen(port, '0.0.0.0', () => {
        console.log(`🚨 Emergency server running on port ${port}`);
      });
    })
    .catch(err => {
      console.error('Failed to create even the emergency server:', err);
      console.error('Server cannot start in any form. Please check your deployment configuration.');
      process.exit(1);
    });
}