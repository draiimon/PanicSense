/**
 * Deployment Readiness Check Script for PanicSense
 * This script checks if your project is ready for deployment on Render.com
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('========================================');
console.log('PanicSense Deployment Readiness Check');
console.log('========================================');

// Check for required files
const requiredFiles = [
  'render-start.js',
  'Procfile',
  '.nvmrc',
  'render.json'
];

let allFilesExist = true;
console.log('\n📋 Checking for required deployment files:');

for (const file of requiredFiles) {
  const exists = fs.existsSync(path.join(__dirname, file));
  console.log(`${exists ? '✅' : '❌'} ${file} ${exists ? 'exists' : 'is missing'}`);
  if (!exists) allFilesExist = false;
}

// Check build commands
console.log('\n🔨 Checking build configuration:');
try {
  const packageJson = JSON.parse(fs.readFileSync(path.join(__dirname, 'package.json'), 'utf8'));
  
  const hasBuildScript = packageJson.scripts && packageJson.scripts.build;
  console.log(`${hasBuildScript ? '✅' : '❌'} Build script ${hasBuildScript ? 'exists' : 'is missing'}`);
  
  const hasStartScript = packageJson.scripts && (packageJson.scripts['start:prod'] || packageJson.scripts.start);
  console.log(`${hasStartScript ? '✅' : '❌'} Production start script ${hasStartScript ? 'exists' : 'is missing'}`);
  
  if (!hasBuildScript || !hasStartScript) allFilesExist = false;
} catch (err) {
  console.log('❌ Error reading package.json');
  allFilesExist = false;
}

// Check if render-start.js is properly configured
try {
  const renderStartContent = fs.readFileSync(path.join(__dirname, 'render-start.js'), 'utf8');
  const hasProperImport = renderStartContent.includes('import(');
  console.log(`${hasProperImport ? '✅' : '❌'} render-start.js ${hasProperImport ? 'looks good' : 'needs review'}`);
  if (!hasProperImport) allFilesExist = false;
} catch (err) {
  console.log('❌ Error reading render-start.js');
  allFilesExist = false;
}

// Final readiness assessment
console.log('\n🔍 Final assessment:');
if (allFilesExist) {
  console.log('✅ Your project appears to be ready for deployment on Render.com!');
  console.log('📝 Next steps:');
  console.log('  1. Push your code to a Git repository');
  console.log('  2. Create a new Web Service on Render.com');
  console.log('  3. Use the settings provided in render-deploy-guide.md');
} else {
  console.log('❌ Some issues need to be addressed before deployment.');
  console.log('📝 Review the warnings above and fix any missing files or configurations.');
}

console.log('\n========================================');
console.log('Check complete!');
console.log('========================================');