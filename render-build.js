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
    packageJson.scripts.renderBuild = "tsc --project tsconfig.render.json";
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
    "exclude": ["node_modules", "build", "client", "**/*.test.ts"],
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

// Step 3: Run the build
try {
  console.log('Running Render-specific build...');
  execSync('npm run renderBuild', { stdio: 'inherit' });
  console.log('Build completed successfully');
} catch (err) {
  console.error('Build failed:', err);
  process.exit(1);
}

// Step 4: Create a CommonJS bootstrap file
try {
  console.log('Creating CommonJS bootstrap file...');
  const bootstrapContent = `// CommonJS bootstrap for Render deployment
const { app, server } = require('./dist/server/index.js');

// Export for Render to use
module.exports = { app, server };
`;
  
  fs.writeFileSync(path.join(process.cwd(), 'dist', 'bootstrap.js'), bootstrapContent);
  console.log('Successfully created bootstrap.js');
} catch (err) {
  console.error('Failed to create bootstrap.js:', err);
  process.exit(1);
}

console.log('Render build process completed successfully!');