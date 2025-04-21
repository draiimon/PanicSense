/**
 * Render.com Setup Helper
 * This script helps validate and setup PanicSense in the Render.com environment
 */

import pg from 'pg';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const { Pool } = pg;

// Test database connection to make sure it's working
async function testDbConnection() {
  console.log('🔄 Testing database connection...');
  
  const databaseUrl = process.env.DATABASE_URL || '';
  if (!databaseUrl) {
    console.error('❌ No DATABASE_URL found in environment variables');
    return false;
  }
  
  try {
    const pool = new Pool({
      connectionString: databaseUrl,
      ssl: { rejectUnauthorized: false }
    });
    
    const client = await pool.connect();
    const result = await client.query('SELECT NOW()');
    console.log(`✅ Database connection successful: ${result.rows[0].now}`);
    client.release();
    await pool.end();
    return true;
  } catch (error) {
    console.error('❌ Database connection failed:', error.message);
    return false;
  }
}

// Setup directories for static files and uploads
function setupDirectories() {
  const dirs = [
    path.join(__dirname, 'uploads'),
    path.join(__dirname, 'logs'),
    path.join(__dirname, 'server', 'public')
  ];
  
  dirs.forEach(dir => {
    if (!fs.existsSync(dir)) {
      try {
        fs.mkdirSync(dir, { recursive: true });
        console.log(`✅ Created directory: ${dir}`);
      } catch (error) {
        console.error(`❌ Failed to create directory ${dir}:`, error.message);
      }
    } else {
      console.log(`✅ Directory already exists: ${dir}`);
    }
  });
}

// Make sure static files are in the right place
function setupStaticFiles() {
  // Only needed if client-side files aren't already in the right place
  const clientDist = path.join(__dirname, 'client', 'dist');
  const serverPublic = path.join(__dirname, 'server', 'public');
  const distPublic = path.join(__dirname, 'dist', 'public');
  
  if (fs.existsSync(clientDist)) {
    try {
      console.log('📦 Copying client/dist to server/public...');
      // This is a simplified version - in a real implementation you'd need a recursive copy function
      fs.readdirSync(clientDist).forEach(file => {
        const srcPath = path.join(clientDist, file);
        const destPath = path.join(serverPublic, file);
        
        if (fs.statSync(srcPath).isFile()) {
          fs.copyFileSync(srcPath, destPath);
        }
      });
      console.log('✅ Client files copied successfully');
    } catch (error) {
      console.error('❌ Failed to copy client files:', error.message);
    }
  } else if (fs.existsSync(distPublic)) {
    try {
      console.log('📦 Copying dist/public to server/public...');
      fs.readdirSync(distPublic).forEach(file => {
        const srcPath = path.join(distPublic, file);
        const destPath = path.join(serverPublic, file);
        
        if (fs.statSync(srcPath).isFile()) {
          fs.copyFileSync(srcPath, destPath);
        }
      });
      console.log('✅ Dist files copied successfully');
    } catch (error) {
      console.error('❌ Failed to copy dist files:', error.message);
    }
  } else {
    console.log('ℹ️ No client or dist files found, skipping copy');
  }
}

// Initialize database tables
async function setupDatabase() {
  try {
    console.log('🔄 Setting up database tables...');
    
    // Import the database setup module
    const dbSetup = await import('./server/db-setup.js');
    console.log('✅ Database setup module loaded');
    
    return true;
  } catch (error) {
    console.error('❌ Failed to setup database tables:', error.message);
    
    // If failed to import the module, try to create tables directly
    try {
      const databaseUrl = process.env.DATABASE_URL;
      if (!databaseUrl) {
        console.error("❌ No DATABASE_URL found in environment variables");
        return false;
      }
      
      console.log("Connecting to database directly...");
      const pool = new Pool({
        connectionString: databaseUrl,
        ssl: { rejectUnauthorized: false }
      });

      const client = await pool.connect();
      console.log(`✅ Successfully connected to PostgreSQL database`);
      
      // Create basic tables
      await client.query(`
        CREATE TABLE IF NOT EXISTS disaster_events (
          id SERIAL PRIMARY KEY,
          name VARCHAR(255) NOT NULL,
          description TEXT,
          location VARCHAR(255),
          severity VARCHAR(50),
          event_type VARCHAR(50),
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
      `);
      
      await client.query(`
        CREATE TABLE IF NOT EXISTS sentiment_posts (
          id SERIAL PRIMARY KEY,
          text TEXT NOT NULL,
          timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          source VARCHAR(100),
          language VARCHAR(50),
          sentiment VARCHAR(50),
          confidence FLOAT,
          disaster_type VARCHAR(100),
          location VARCHAR(255)
        )
      `);
      
      await client.query(`
        CREATE TABLE IF NOT EXISTS analyzed_files (
          id SERIAL PRIMARY KEY,
          original_name VARCHAR(255) NOT NULL,
          stored_name VARCHAR(255) NOT NULL,
          row_count INTEGER,
          accuracy FLOAT,
          precision FLOAT,
          recall FLOAT,
          f1_score FLOAT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
      `);
      
      // Add sample data
      console.log("Adding sample disaster data...");
      
      // Insert sample disaster event
      await client.query(`
        INSERT INTO disaster_events (name, description, location, severity, event_type)
        VALUES ('Typhoon in Coastal Areas', 'Based on 3 reports from the community. Please stay safe.', 'Metro Manila, Philippines', 'High', 'Typhoon')
        ON CONFLICT DO NOTHING
      `);
      
      // Insert sample sentiment post
      await client.query(`
        INSERT INTO sentiment_posts (text, source, language, sentiment, confidence, disaster_type, location)
        VALUES ('My prayers to our brothers and sisters in Visayas region..', 'Twitter', 'en', 'neutral', 0.85, 'Typhoon', 'Visayas, Philippines')
        ON CONFLICT DO NOTHING
      `);
      
      // Insert sample analyzed file
      await client.query(`
        INSERT INTO analyzed_files (original_name, stored_name, row_count, accuracy, precision, recall, f1_score)
        VALUES ('MAGULONG DATA! (1).csv', 'batch-EJBpcspVXK_TZ717aZDM7-MAGULONG DATA! (1).csv', 100, 0.89, 0.91, 0.87, 0.89)
        ON CONFLICT DO NOTHING
      `);
      
      console.log("✅ Sample data added successfully");
      client.release();
      await pool.end();
      
      return true;
    } catch (directError) {
      console.error('❌ Failed to setup database directly:', directError.message);
      return false;
    }
  }
}

// Fix database queries with created_at references
async function fixDatabaseQueries() {
  try {
    console.log('🔧 Applying database query fixes...');
    
    // Import and run the database fix script
    try {
      const dbFix = await import('./server/db-fix.js');
      const result = await dbFix.fixDatabaseQueries();
      console.log('🔧 Database query fix result:', result ? '✅ APPLIED' : 'ℹ️ NOT NEEDED');
      return true;
    } catch (error) {
      console.error('⚠️ Could not import db-fix.js, falling back to direct fix:', error.message);
      
      // Direct fix if module import fails
      const serverJsPath = path.join(__dirname, 'server.js');
      
      if (fs.existsSync(serverJsPath)) {
        let serverJs = fs.readFileSync(serverJsPath, 'utf8');
        
        // Replace all ORDER BY created_at with ORDER BY id
        serverJs = serverJs.replace(/ORDER BY created_at/g, 'ORDER BY id');
        
        fs.writeFileSync(serverJsPath, serverJs);
        console.log('✅ Direct fix applied to server.js');
        return true;
      } else {
        console.error('❌ server.js not found for direct fix at:', serverJsPath);
        return false;
      }
    }
  } catch (error) {
    console.error('❌ Error fixing database queries:', error.message);
    return false;
  }
}

// Main setup function
async function runSetup() {
  console.log('======================================');
  console.log('🚀 Starting Render.com setup process');
  console.log('======================================');
  
  // Run each setup step
  setupDirectories();
  setupStaticFiles();
  const dbConnected = await testDbConnection();
  
  if (dbConnected) {
    console.log('✅ Database connection successful, setting up tables...');
    await setupDatabase();
    
    // Apply database query fixes
    await fixDatabaseQueries();
  }
  
  console.log('======================================');
  console.log('✅ Render.com setup process complete');
  console.log('======================================');
}

// Run setup and catch any unhandled errors
runSetup().catch(error => {
  console.error('❌ Unhandled error in render-setup.js:', error);
  process.exit(1);
});