/**
 * Static Build Script
 * This builds the client-side app without using Vite for better compatibility with Render
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// For ES modules compatibility
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('Starting static build process...');

// Define directories
const clientDir = path.join(__dirname, 'client');
const srcDir = path.join(clientDir, 'src');
const distDir = path.join(__dirname, 'dist', 'public');
const publicDir = path.join(clientDir, 'public');

// Make sure dist directory exists
if (!fs.existsSync(path.join(__dirname, 'dist'))) {
  fs.mkdirSync(path.join(__dirname, 'dist'));
}

if (!fs.existsSync(distDir)) {
  fs.mkdirSync(distDir);
}

try {
  // First build the client using esbuild
  console.log('Building client with esbuild...');
  
  execSync(`npx esbuild ${path.join(srcDir, 'main.tsx')} \
    --bundle \
    --minify \
    --sourcemap \
    --target=es2020 \
    --format=esm \
    --jsx=automatic \
    --outdir=${distDir}`, { stdio: 'inherit' });
  
  // Copy the HTML file and replace the script source
  console.log('Copying and updating index.html...');
  let htmlContent = fs.readFileSync(path.join(clientDir, 'index.html'), 'utf-8');
  
  // Update script source
  htmlContent = htmlContent.replace(
    '<script type="module" src="/src/main.tsx"></script>',
    '<script type="module" src="/main.js"></script>'
  );
  
  fs.writeFileSync(path.join(distDir, 'index.html'), htmlContent);
  
  // Copy public files if they exist
  if (fs.existsSync(publicDir)) {
    console.log('Copying public files...');
    const files = fs.readdirSync(publicDir);
    
    for (const file of files) {
      const srcFile = path.join(publicDir, file);
      const destFile = path.join(distDir, file);
      
      if (fs.statSync(srcFile).isDirectory()) {
        // Use recursive copy for directories
        execSync(`cp -r ${srcFile} ${distDir}`);
      } else {
        fs.copyFileSync(srcFile, destFile);
      }
    }
  }
  
  // Copy CSS files
  if (fs.existsSync(path.join(srcDir, 'index.css'))) {
    console.log('Copying CSS files...');
    fs.copyFileSync(
      path.join(srcDir, 'index.css'),
      path.join(distDir, 'index.css')
    );
  }
  
  console.log('Static build completed successfully!');
} catch (error) {
  console.error('Build failed:', error);
  process.exit(1);
}