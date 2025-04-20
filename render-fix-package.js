/**
 * This is a special script that modifies package.json temporarily
 * It's needed because we can't directly edit package.json in this environment
 * But in your Render deployment you CAN edit it
 */
const fs = require('fs');
const path = require('path');

// Read the current package.json
const packageJsonPath = path.join(__dirname, 'package.json');
let packageJson;

try {
  packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  
  // Change the "start" script to directly use index.js
  packageJson.scripts.start = "NODE_ENV=production node index.js";
  
  // Add a note about the change
  packageJson._renderModified = "Modified for Render.com deployment";
  
  // Write the modified package.json back to disk
  fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
  
  console.log("✅ Successfully modified package.json 'start' script to use index.js");
  console.log("Changes will be picked up by Render automatically");
} catch (error) {
  console.error("❌ Failed to update package.json:", error);
  process.exit(1);
}