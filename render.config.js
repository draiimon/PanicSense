/**
 * Render.com deployment configuration
 * This file is used specifically for Render platform deployments
 */

const config = {
  // Build command that works on Render
  buildCommand: "npx next build && npx esbuild server/index-wrapper.js --platform=node --packages=external --bundle --format=esm --outdir=dist",
  
  // Start command that works on Render
  startCommand: "NODE_ENV=production node dist/index-wrapper.js",
  
  // Environment variables needed for deployment
  environmentVariables: {
    NODE_ENV: "production",
    PORT: "10000" // Render sets this automatically, but we have a fallback
  }
};

export default config;