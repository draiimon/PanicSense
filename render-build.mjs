// This is a special build script for Render deployment using ESM
// Its purpose is to create an ESM-compatible build that works with top-level await

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('Starting Render ESM build process...');

// Step 1: Ensure package.json has the correct configuration
try {
  console.log('Configuring package.json for ESM...');
  const packageJsonPath = path.join(process.cwd(), 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  
  // Make sure type is set to "module"
  if (!packageJson.type || packageJson.type !== 'module') {
    packageJson.type = 'module';
    console.log('Set "type": "module" in package.json');
  }
  
  // Make sure the build script is correct
  packageJson.scripts.renderBuild = "vite build && esbuild server/index-wrapper.ts --platform=node --packages=external --bundle --format=esm --outdir=dist";
  
  fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
  console.log('Successfully updated package.json for ESM');
} catch (err) {
  console.error('Failed to modify package.json:', err);
  process.exit(1);
}

// Step 2: Create a special tsconfig for Render
try {
  console.log('Creating special tsconfig for Render...');
  const tsConfig = {
    "include": ["server/**/*", "shared/**/*"],
    "exclude": ["node_modules", "build", "client/src", "**/*.test.ts"],
    "compilerOptions": {
      "target": "ES2022",
      "module": "NodeNext",
      "moduleResolution": "NodeNext",
      "esModuleInterop": true,
      "outDir": "./dist",
      "rootDir": "./",
      "strict": true,
      "skipLibCheck": true,
      "forceConsistentCasingInFileNames": true,
      "baseUrl": ".",
      "allowSyntheticDefaultImports": true,
      "resolveJsonModule": true,
      "paths": {
        "@shared/*": ["./shared/*"]
      }
    }
  };

  fs.writeFileSync(
    path.join(process.cwd(), 'tsconfig.render.json'), 
    JSON.stringify(tsConfig, null, 2)
  );
  console.log('Successfully created tsconfig.render.json');
} catch (err) {
  console.error('Failed to create tsconfig.render.json:', err);
  process.exit(1);
}

// Step 3: Create a proper index file for production that properly handles top-level await
try {
  console.log('Creating production index file...');
  const wrapperContent = `/**
 * This is a special production index file
 * It ensures top-level await is properly handled in ESM
 */

import express from 'express';
import { Server } from 'http';
import { WebSocketServer } from 'ws';
import { registerRoutes } from './routes.js';
import { simpleDbFix } from './db-simple-fix.js';
import { serveStatic } from './vite.js';

// Create Express server
const app = express();
let server;

// This function encapsulates all the async initialization
async function initializeServer() {
  console.log('========================================');
  console.log(\`Starting server initialization at: \${new Date().toISOString()}\`);
  console.log('========================================');

  // Apply database fixes for production
  console.log('Running database fixes...');
  await simpleDbFix();

  // Basic middleware
  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ extended: false, limit: '50mb' }));

  // Register routes
  server = await registerRoutes(app);
  console.log('Routes registered successfully');
  
  // Error handling middleware
  app.use((err, _req, res, _next) => {
    console.error('Server error:', err);
    res.status(500).json({ 
      error: true,
      message: err.message || 'Internal Server Error',
      timestamp: new Date().toISOString()
    });
  });
  
  // We're in production
  console.log('Running in production mode, serving static files...');
  serveStatic(app);
  
  // Get port from environment with fallback
  const port = parseInt(process.env.PORT || '10000', 10);
  console.log(\`Attempting to listen on port \${port}...\`);
  
  // Start server
  return new Promise((resolve) => {
    server.listen(port, '0.0.0.0', () => {
      console.log(\`========================================\`);
      console.log(\`🚀 Server running on port \${port}\`);
      console.log(\`Server listening at: http://0.0.0.0:\${port}\`);
      console.log(\`Server ready at: \${new Date().toISOString()}\`);
      console.log(\`========================================\`);
      resolve({ success: true });
    });
  });
}

// Setup graceful shutdown
function setupShutdownHandlers() {
  process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down gracefully');
    server?.close(() => {
      console.log('HTTP server closed');
      process.exit(0);
    });
    
    // Force close after 10 seconds
    setTimeout(() => {
      console.log('Forcing shutdown after timeout');
      process.exit(1);
    }, 10000);
  });
  
  process.on('SIGINT', () => {
    console.log('SIGINT received, shutting down gracefully');
    server?.close(() => {
      console.log('HTTP server closed');
      process.exit(0);
    });
    
    // Force close after 10 seconds
    setTimeout(() => {
      console.log('Forcing shutdown after timeout');
      process.exit(1);
    }, 10000);
  });
}

// This is the main entry point - properly wrapped to avoid top-level await issues
async function main() {
  try {
    setupShutdownHandlers();
    await initializeServer();
    return { success: true };
  } catch (error) {
    console.error('Fatal error during server initialization:', error);
    return { success: false, error };
  }
}

// Execute the main function and handle any errors
main().catch(err => {
  console.error('Unhandled error in main function:', err);
  process.exit(1);
});

export { app, server };
`;
  
  fs.writeFileSync(path.join(process.cwd(), 'server', 'production.ts'), wrapperContent);
  console.log('Successfully created production index file');
} catch (err) {
  console.error('Failed to create production index file:', err);
  process.exit(1);
}

// Step 4: Update the build command to use our production file
try {
  console.log('Running custom build for Render...');
  
  // Create the build script to handle ESM modules with top-level await
  const buildScript = `#!/usr/bin/env node

// First build the frontend
console.log('Building frontend...');
try {
  execSync('vite build', { stdio: 'inherit' });
} catch (e) {
  console.error('Frontend build failed:', e);
  process.exit(1);
}

// Then build the backend
console.log('Building backend...');
try {
  execSync('esbuild server/production.ts --platform=node --packages=external --bundle --format=esm --outfile=dist/index.js', { stdio: 'inherit' });
} catch (e) {
  console.error('Backend build failed:', e);
  process.exit(1);
}

console.log('Build completed successfully');
`;

  fs.writeFileSync(path.join(process.cwd(), 'build-render.js'), buildScript);
  fs.chmodSync(path.join(process.cwd(), 'build-render.js'), '755');
  
  // Create a package.json update
  const packageJsonPath = path.join(process.cwd(), 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  packageJson.scripts.buildRender = "node build-render.js";
  fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
  
  console.log('Build script created successfully');
} catch (err) {
  console.error('Failed to setup build script:', err);
  process.exit(1);
}

console.log('Render build process setup completed successfully!');