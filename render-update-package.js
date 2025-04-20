/**
 * This script is for Render.com deployment only
 * It modifies package.json at deployment time to fix ESM/CommonJS compatibility issues
 */

const fs = require('fs');
const path = require('path');

console.log('Starting package.json update for Render deployment...');

// Read the current package.json
try {
  const packageJsonPath = path.join(__dirname, 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  
  // Remove type:module so Node.js treats index.js as CommonJS
  delete packageJson.type;
  
  // Update scripts to use the CommonJS index.js file
  packageJson.scripts.start = 'NODE_ENV=production node index.js';
  packageJson.scripts.build = 'vite build';
  
  // Add a note about the changes
  packageJson._renderDeploymentUpdated = true;
  packageJson._renderDeploymentNote = 'Modified for CommonJS compatibility on Render.com';
  
  // Write the updated package.json
  fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
  
  console.log('✅ Successfully updated package.json for Render deployment');
} catch (err) {
  console.error('❌ Error updating package.json:', err);
  process.exit(1);
}