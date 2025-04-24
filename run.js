/**
 * Special runner script for Render deployment
 * This is a simplified entry point that ensures the correct files are loaded
 */

// Import required modules
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('========================================');
console.log(`🚀 PanicSense Starter for Render Deployment`);
console.log(`Starting at: ${new Date().toISOString()}`);
console.log('========================================');

// Try multiple possible entry points
const possibleEntryPoints = [
  './server/index-wrapper.js',
  './server/index.js',
  './index.js',
  './server.js'
];

let entryPointFound = false;

for (const entryPoint of possibleEntryPoints) {
  console.log(`Checking if entry point exists: ${entryPoint}`);
  
  if (existsSync(join(__dirname, entryPoint))) {
    console.log(`✅ Found entry point: ${entryPoint}`);
    console.log(`Starting server from: ${entryPoint}`);
    
    // Dynamic import to start the server
    import(entryPoint)
      .then(() => {
        console.log(`Server started successfully via ${entryPoint}`);
      })
      .catch(err => {
        console.error(`❌ Failed to start server from ${entryPoint}:`, err);
        process.exit(1);
      });
    
    entryPointFound = true;
    break;
  }
}

if (!entryPointFound) {
  console.error('❌ No valid entry point found! Tried:', possibleEntryPoints);
  console.error('Creating emergency server...');
  
  // Create an emergency basic server if no entry point is found
  import('express')
    .then(({ default: express }) => {
      const app = express();
      
      app.get('/', (_req, res) => {
        res.send(`
          <html>
            <head>
              <title>PanicSense - Emergency Mode</title>
              <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; padding: 20px; line-height: 1.6; }
                header { background-color: #e74c3c; color: white; padding: 20px; border-radius: 5px; }
                h1 { margin: 0; }
                .content { margin-top: 20px; }
                .api-box { background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #e74c3c; margin: 20px 0; }
              </style>
            </head>
            <body>
              <header>
                <h1>PanicSense - Emergency Mode</h1>
              </header>
              <div class="content">
                <p>The PanicSense server is running in emergency mode.</p>
                <div class="api-box">
                  <h3>Limited Functionality</h3>
                  <p>Only basic API endpoints are available.</p>
                  <p>For full functionality, please check the server configuration.</p>
                </div>
              </div>
            </body>
          </html>
        `);
      });
      
      app.get('/api/health', (_req, res) => {
        res.json({ status: 'ok', mode: 'emergency' });
      });
      
      const port = parseInt(process.env.PORT || '10000');
      app.listen(port, '0.0.0.0', () => {
        console.log(`🚨 Emergency server running on port ${port}`);
      });
    })
    .catch(err => {
      console.error('Failed to create emergency server:', err);
      process.exit(1);
    });
}