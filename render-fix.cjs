/**
 * Special fix script for Render.com deployment
 * 
 * This script is meant to be executed by Render during deployment
 * It adjusts package.json to prevent TypeScript compilation
 * and ensures we use our plain JS server directly
 */

const fs = require('fs');
const path = require('path');

// Log message
console.log("Running render-fix.cjs - Adjusting package.json for Render deployment");

try {
  // Read the current package.json
  const packageJsonPath = path.join(process.cwd(), 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));

  // Modify the start script to use our plain JS server
  packageJson.scripts.start = "NODE_ENV=production node server/production-server.js";
  
  // Add a comment explaining the change
  packageJson.renderDeployment = {
    comment: "This file was modified by render-fix.cjs to avoid TypeScript compilation issues",
    originalStart: "NODE_ENV=production node dist/index.js"
  };

  // Write back the modified package.json
  fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
  
  console.log("✅ Successfully modified package.json for Render deployment");
  console.log("Start script now points directly to server/production-server.js");
} catch (error) {
  console.error("❌ Error modifying package.json:", error);
  process.exit(1);
}