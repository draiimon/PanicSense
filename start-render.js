/**
 * This is a simple starter script that redirects to start-render.cjs
 * It's needed because Render sometimes has issues with .cjs files directly
 */

// Set production environment
process.env.NODE_ENV = 'production';

// Simple log about what's happening
console.log('============================');
console.log('🚀 STARTING PANICSENSE ON RENDER');
console.log('📂 Using start-render.cjs script');
console.log('============================');

// Load the CJS version of the start script
require('./start-render.cjs');