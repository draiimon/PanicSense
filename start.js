/**
 * PanicSense Render.com Free Tier - Emergency Start Script
 * 
 * This is a simplified fallback startup script for PanicSense when deployed on Render.com.
 * If the regular build process fails, Render can use this script as an alternative startup command.
 */

console.log('🚨 Starting PanicSense in EMERGENCY MODE');
console.log('This is a fallback script for Render.com free tier deployment');

// Force production mode
process.env.NODE_ENV = 'production';

// Try different startup methods
try {
  console.log('Attempting to start server using server/index-wrapper.js...');
  require('./server/index-wrapper.js');
} catch (error) {
  console.error('Failed to start using index-wrapper.js:', error.message);
  
  try {
    console.log('Attempting to start server using server/index.js...');
    require('./server/index.js');
  } catch (secondError) {
    console.error('Failed to start using index.js:', secondError.message);
    
    try {
      console.log('Attempting final fallback with server.js...');
      require('./server.js');
    } catch (finalError) {
      console.error('CRITICAL FAILURE: All startup methods failed!');
      console.error('Please check your deployment configuration.');
      console.error('Final error:', finalError);
      process.exit(1);
    }
  }
}