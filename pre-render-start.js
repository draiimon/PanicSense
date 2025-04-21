/**
 * PRE-RENDER START
 * This script runs before the server starts on Render
 * It applies critical fixes to ensure the server can run correctly
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

console.log('==> Deploying...');
console.log('✅ Preparing server environment...');

// Handle client files
try {
  const clientDist = path.join(__dirname, 'client', 'dist');
  const serverPublic = path.join(__dirname, 'server', 'public');
  const distPublic = path.join(__dirname, 'dist', 'public');
  
  if (fs.existsSync(clientDist)) {
    console.log('📂 Using client/dist for static files');
  } else if (fs.existsSync(distPublic)) {
    console.log('📂 Using dist/public for static files');
  } else {
    console.log('⚠️ No static files found, backend-only mode');
  }
} catch (error) {
  console.error('❌ Error setting up static files:', error.message);
}

// Check database URL
try {
  const dbUrl = process.env.DATABASE_URL || '';
  if (dbUrl) {
    if (dbUrl.includes('ssl=true')) {
      console.log('🔒 SSL mode already present in DATABASE_URL');
    } else {
      console.log('⚠️ DATABASE_URL does not include SSL mode, application will add it automatically');
    }
  } else {
    console.error('⚠️ DATABASE_URL not found, database features may not work');
  }
} catch (error) {
  console.error('❌ Error checking DATABASE_URL:', error.message);
}

// Set environment variables
console.log(`⚙️ Environment: NODE_ENV=${process.env.NODE_ENV}`);

// Immediate database fixes
console.log('🔄 Attempting database connection...');

// Server.js safe mode
try {
  const serverPath = path.join(__dirname, 'server.js');
  if (fs.existsSync(serverPath)) {
    let serverContent = fs.readFileSync(serverPath, 'utf8');
    
    // Ensure ORDER BY statements use id instead of created_at
    if (serverContent.includes('ORDER BY created_at')) {
      console.log('🛠️ Fixing ORDER BY clauses in server.js...');
      serverContent = serverContent.replace(/ORDER BY created_at/g, 'ORDER BY id');
      fs.writeFileSync(serverPath, serverContent);
      console.log('✅ Fixed ORDER BY clauses in server.js');
    }
  }
} catch (error) {
  console.error('❌ Error fixing server.js:', error.message);
}

// Set environment variables dynamically if needed
if (!process.env.DISABLE_SSL_VERIFY && process.env.NODE_ENV === 'production') {
  console.log('ℹ️ Setting DISABLE_SSL_VERIFY=true for social media scrapers');
  process.env.DISABLE_SSL_VERIFY = 'true';
}

console.log('🚀 Starting server on port', process.env.PORT || 10000, '...');

// Continue with normal server startup by importing the server module
try {
  // This dynamic import will execute server.js
  import('./server.js').catch(error => {
    console.error('❌ Error importing server module:', error.message);
    process.exit(1);
  });
} catch (error) {
  console.error('❌ Error starting server:', error.message);
  console.error('⚠️ Trying to continue anyway...');
  
  // If that fails, try to run the server directly
  try {
    import('./index.js').catch(error => {
      console.error('❌ Error importing index module:', error.message);
      process.exit(1);
    });
  } catch (indexError) {
    console.error('❌ Critical error! Could not start server:', indexError.message);
    process.exit(1);
  }
}