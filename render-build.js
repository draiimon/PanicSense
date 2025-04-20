// This is a special build script for Render deployment
// Its purpose is to create a CommonJS-compatible build

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('Starting Render build process...');

// Step 1: Modify package.json to use CommonJS
try {
  console.log('Converting package.json to CommonJS...');
  const packageJsonPath = path.join(process.cwd(), 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  
  // Remove "type": "module" if it exists
  if (packageJson.type === 'module') {
    delete packageJson.type;
    console.log('Removed "type": "module" from package.json');
  }
  
  // Add special render build script if it doesn't exist
  if (!packageJson.scripts.renderBuild) {
    packageJson.scripts.renderBuild = "tsc --project tsconfig.render.json && cp -r client/public dist/client/";
  }
  
  fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
  console.log('Successfully updated package.json for CommonJS');
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
      "target": "ES2018",
      "module": "CommonJS",
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

// Step 3: Rename server/index-wrapper.ts to server/index.ts temporarily
try {
  console.log('Temporarily swapping index-wrapper.ts to index.ts for build...');
  // First backup original index.ts
  fs.copyFileSync(
    path.join(process.cwd(), 'server', 'index.ts'),
    path.join(process.cwd(), 'server', 'index.ts.bak')
  );
  
  // Then copy index-wrapper.ts to index.ts
  fs.copyFileSync(
    path.join(process.cwd(), 'server', 'index-wrapper.ts'),
    path.join(process.cwd(), 'server', 'index.ts')
  );
  
  console.log('Successfully swapped index files');
} catch (err) {
  console.error('Failed to swap index files:', err);
  process.exit(1);
}

// Step 4: Run the build
try {
  console.log('Running Render-specific build...');
  execSync('npm run renderBuild', { stdio: 'inherit' });
  console.log('Build completed successfully');
} catch (err) {
  console.error('Build failed:', err);
  process.exit(1);
} finally {
  // Restore original index.ts
  try {
    fs.copyFileSync(
      path.join(process.cwd(), 'server', 'index.ts.bak'),
      path.join(process.cwd(), 'server', 'index.ts')
    );
    fs.unlinkSync(path.join(process.cwd(), 'server', 'index.ts.bak'));
    console.log('Successfully restored original index.ts');
  } catch (restoreErr) {
    console.error('Warning: Failed to restore original index.ts:', restoreErr);
  }
}

// Step 5: Create a production entry point
try {
  console.log('Creating production entry point...');
  const entryContent = `// Production entry point - no top-level await
// This file was auto-generated during the build process

console.log('Starting production server (CommonJS version)');

// Use require syntax for CommonJS compatibility
const serverModule = require('./server/index.js');

// Export the app and server for Render to use
module.exports = {
  app: serverModule.app,
  server: serverModule.server
};
`;
  
  fs.writeFileSync(path.join(process.cwd(), 'dist', 'index.js'), entryContent);
  console.log('Successfully created production entry point');
} catch (err) {
  console.error('Failed to create production entry point:', err);
  process.exit(1);
}

console.log('Render build process completed successfully!');