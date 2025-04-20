// This is a special build script for Render deployment
// Its purpose is to create an ESM-compatible build that works with top-level await

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('Starting Render build process...');

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
      "module": "ESNext",
      "moduleResolution": "node",
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

// Step 3: Create a proper index-wrapper.ts that handles top-level await
try {
  console.log('Creating improved index-wrapper.ts');
  const wrapperContent = `/**
 * This is a special wrapper for production deployment
 * It ensures top-level await is properly handled in ESM
 */

import express from 'express';
import { Server } from 'http';
import { log } from './vite.js';
import { simpleDbFix } from './db-simple-fix.js';

// Create Express server
const app = express();
let server;

// Wrap the server initialization in an async IIFE
async function init_index() {
  try {
    console.log('========================================');
    console.log(\`Starting server initialization at: \${new Date().toISOString()}\`);
    console.log('========================================');

    // Apply database fixes
    console.log('Running database fixes...');
    await simpleDbFix();

    // Basic middleware
    app.use(express.json({ limit: '50mb' }));
    app.use(express.urlencoded({ extended: false, limit: '50mb' }));

    // Import dynamic modules - must be dynamic imports to avoid top-level await issues
    const { registerRoutes } = await import('./routes.js');
    const { serveStatic } = await import('./vite.js');
    
    // Register routes 
    server = await registerRoutes(app);
    console.log('Routes registered successfully');
    
    // Handle errors
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
    
    // Start server
    const port = parseInt(process.env.PORT || '10000', 10);
    console.log(\`Attempting to listen on port \${port}...\`);
    
    server.listen(port, '0.0.0.0', () => {
      console.log(\`========================================\`);
      log(\`🚀 Server running on port \${port}\`);
      console.log(\`Server listening at: http://0.0.0.0:\${port}\`);
      console.log(\`Server ready at: \${new Date().toISOString()}\`);
      console.log(\`========================================\`);
    });

    // Setup shutdown handlers
    process.on('SIGTERM', () => {
      console.log('SIGTERM signal received: closing server');
      server.close(() => {
        console.log('Server closed');
        process.exit(0);
      });
    });
    
    process.on('SIGINT', () => {
      console.log('SIGINT signal received: closing server');
      server.close(() => {
        console.log('Server closed');
        process.exit(0);
      });
    });

    return { success: true };
  } catch (error) {
    console.error('Failed to start server:', error);
    return { success: false, error };
  }
}

// Initialize the server with await properly contained in a function
const initPromise = init_index();

// Export for production
export { app, server, initPromise };
`;
  
  fs.writeFileSync(path.join(process.cwd(), 'server', 'index-wrapper-prod.ts'), wrapperContent);
  console.log('Successfully created improved wrapper');
} catch (err) {
  console.error('Failed to create improved wrapper:', err);
  process.exit(1);
}

// Step 4: Run the build with the improved wrapper
try {
  console.log('Running Render-specific build...');
  // Copy the improved wrapper to the index-wrapper.ts file
  fs.copyFileSync(
    path.join(process.cwd(), 'server', 'index-wrapper-prod.ts'),
    path.join(process.cwd(), 'server', 'index-wrapper.ts')
  );
  
  // Create a special entry point for production
  const productionEntryPoint = `// Production entry point for Render
// ESM version with proper top-level await handling

console.log('Starting production server (ESM version)');

// Import the server
import { initPromise } from './server/index-wrapper.js';

// Wait for initialization
initPromise.then(result => {
  if (result.success) {
    console.log('Server initialized successfully');
  } else {
    console.error('Server initialization failed:', result.error);
    process.exit(1);
  }
}).catch(err => {
  console.error('Fatal error during initialization:', err);
  process.exit(1);
});
`;

  // Create the dist directory if it doesn't exist
  if (!fs.existsSync(path.join(process.cwd(), 'dist'))) {
    fs.mkdirSync(path.join(process.cwd(), 'dist'));
  }
  
  // Write the production entry point
  fs.writeFileSync(path.join(process.cwd(), 'dist', 'index-prod.js'), productionEntryPoint);
  
  // Run the build
  execSync('npm run renderBuild', { stdio: 'inherit' });
  
  // Copy the entry point to index.js
  fs.copyFileSync(
    path.join(process.cwd(), 'dist', 'index-prod.js'),
    path.join(process.cwd(), 'dist', 'index.js')
  );
  
  console.log('Build completed successfully');
} catch (err) {
  console.error('Build failed:', err);
  process.exit(1);
}

console.log('Render build process completed successfully!');