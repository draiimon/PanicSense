/**
 * PanicSense Render.com - Super Simple Start Script
 * Hindi na kailangan ng Vite, ESBuild o anumang build tools!
 * 
 * Ito ay direct-to-server approach para sa Render free tier
 */

// Set production environment
process.env.NODE_ENV = 'production';

// Show startup message
console.log('🚀 Starting PanicSense (PURE SERVER MODE - NO BUILD NEEDED!)');
console.log('✅ Render Free Tier Compatible Version');

// Try different startup methods
try {
  // Try the regular server file first
  console.log('Loading from server/index-wrapper.js...');
  require('./server/index-wrapper.js');
} catch (error) {
  console.error('First startup method failed, trying backup method...', error.message);

  try {
    // Try direct server file
    console.log('Loading from server/index.js...');
    require('./server/index.js');
  } catch (error2) {
    console.error('Second startup method failed, trying emergency fallback...', error2.message);

    try {
      // Try server.js at root
      console.log('Loading from server.js at root...');
      require('./server.js');
    } catch (error3) {
      console.error('EMERGENCY FALLBACK FAILED!', error3.message);
      
      // Last ditch effort - create an Express server directly
      try {
        console.log('Creating minimal Express server directly...');
        const express = require('express');
        const app = express();
        const port = process.env.PORT || 10000;
        
        // Use public directory if it exists
        if (require('fs').existsSync('./dist/public')) {
          app.use(express.static('./dist/public'));
        } else if (require('fs').existsSync('./public')) {
          app.use(express.static('./public'));
        }
        
        // Basic routes
        app.get('/', (req, res) => {
          res.send('PanicSense API Server is running in emergency mode');
        });
        
        app.get('/api/health', (req, res) => {
          res.json({ status: 'ok', mode: 'emergency' });
        });
        
        // Start server
        app.listen(port, '0.0.0.0', () => {
          console.log(`Emergency server started on port ${port}`);
        });
      } catch (finalError) {
        console.error('FATAL ERROR: All server startup methods failed!');
        console.error(finalError);
        process.exit(1);
      }
    }
  }
}