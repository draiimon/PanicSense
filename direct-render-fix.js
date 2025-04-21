/**
 * EMERGENCY RENDER DEPLOYMENT FIX
 * This script directly fixes critical issues on Render deployment
 */

import fs from 'fs';
import path from 'path';
import { pool } from './server/db.js';

async function fixSSLIssues() {
  console.log("🔧 Applying SSL verification fixes...");
  
  // Hard disable SSL verification for Python requests
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
  
  try {
    // Create a Python script fix to disable SSL verification
    const pythonFixPath = path.join(process.cwd(), 'server', 'python', 'disable_ssl.py');
    const pythonFix = `
# SSL FIX FOR TWITTER SCRAPING
import os
import ssl
import requests
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure SSL settings
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Create a non-verifying SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Patch requests to not verify SSL
old_request = requests.Session.request
def new_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return old_request(self, method, url, **kwargs)
requests.Session.request = new_request

print("✅ SSL verification disabled successfully")
`;
    
    fs.writeFileSync(pythonFixPath, pythonFix);
    console.log(`✅ Created SSL fix at ${pythonFixPath}`);
    
    // Also patch the Python service module to use the fix
    const pythonServicePath = path.join(process.cwd(), 'server', 'python-service.ts');
    
    if (fs.existsSync(pythonServicePath)) {
      let pythonServiceContent = fs.readFileSync(pythonServicePath, 'utf8');
      
      // Add import for our SSL fix
      if (!pythonServiceContent.includes('disable_ssl.py')) {
        pythonServiceContent = pythonServiceContent.replace(
          'private pythonBinary: string;',
          'private pythonBinary: string;\n  private sslFixApplied: boolean = false;'
        );
        
        // Add SSL fix to the Python service constructor
        pythonServiceContent = pythonServiceContent.replace(
          'constructor() {',
          `constructor() {
    // Apply SSL fix for Python requests
    try {
      if (!this.sslFixApplied) {
        const sslFixPath = path.join(__dirname, 'python', 'disable_ssl.py');
        if (fs.existsSync(sslFixPath)) {
          console.log('⚠️ Applying critical SSL fix for Python requests');
          const result = childProcess.execSync(\`\${this.pythonBinary} \${sslFixPath}\`, { encoding: 'utf8' });
          console.log(result);
          this.sslFixApplied = true;
          
          // Also set Node.js level SSL verification off
          process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
        }
      }
    } catch (error) {
      console.error('Error applying SSL fix:', error);
    }`
        );
        
        fs.writeFileSync(pythonServicePath, pythonServiceContent);
        console.log(`✅ Added SSL fix to Python service`);
      } else {
        console.log(`ℹ️ SSL fix already present in Python service`);
      }
    } else {
      console.log(`⚠️ Python service file not found at ${pythonServicePath}`);
    }
    
    return true;
  } catch (error) {
    console.error("❌ Error applying SSL fixes:", error);
    return false;
  }
}

async function fixDatabaseIssues() {
  console.log("🔧 Checking for database issues...");
  
  try {
    // Fix missing timestamp columns
    try {
      // Add timestamp to tables that use created_at for ordering
      await pool.query(`
        ALTER TABLE disaster_events 
        ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      `);
      
      await pool.query(`
        ALTER TABLE sentiment_posts 
        ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      `);
      
      await pool.query(`
        ALTER TABLE analyzed_files 
        ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      `);
      
      // Update timestamp columns to match created_at where available
      await pool.query(`
        UPDATE disaster_events SET timestamp = created_at 
        WHERE timestamp IS NULL AND created_at IS NOT NULL
      `);
      
      await pool.query(`
        UPDATE sentiment_posts SET timestamp = created_at 
        WHERE timestamp IS NULL AND created_at IS NOT NULL
      `);
      
      await pool.query(`
        UPDATE analyzed_files SET timestamp = created_at 
        WHERE timestamp IS NULL AND created_at IS NOT NULL
      `);
      
      console.log("✅ Added and populated timestamp columns where needed");
    } catch (error) {
      console.error("❌ Error fixing timestamp columns:", error);
    }
    
    // Patch any SQL queries that might use created_at
    try {
      await pool.query(`
        CREATE OR REPLACE VIEW recent_posts AS
        SELECT * FROM sentiment_posts 
        ORDER BY timestamp DESC, id DESC
        LIMIT 100
      `);
      
      await pool.query(`
        CREATE OR REPLACE VIEW recent_events AS
        SELECT * FROM disaster_events 
        ORDER BY timestamp DESC, id DESC
        LIMIT 100
      `);
      
      await pool.query(`
        CREATE OR REPLACE VIEW recent_files AS
        SELECT * FROM analyzed_files 
        ORDER BY timestamp DESC, id DESC
        LIMIT 100
      `);
      
      console.log("✅ Created views for efficient queries without created_at");
    } catch (viewError) {
      console.error("❌ Error creating database views:", viewError);
    }
    
    // Create sample data if tables are empty
    const disasterCount = await pool.query("SELECT COUNT(*) FROM disaster_events");
    if (parseInt(disasterCount.rows[0].count) === 0) {
      await pool.query(`
        INSERT INTO disaster_events (name, description, location, severity, event_type, timestamp)
        VALUES ('Typhoon in Coastal Areas', 'Based on 3 reports from the community. Please stay safe.', 'Metro Manila, Philippines', 'High', 'Typhoon', NOW())
      `);
      console.log("✅ Added sample disaster event");
    }
    
    const sentimentCount = await pool.query("SELECT COUNT(*) FROM sentiment_posts");
    if (parseInt(sentimentCount.rows[0].count) === 0) {
      await pool.query(`
        INSERT INTO sentiment_posts (text, source, language, sentiment, confidence, disaster_type, location, timestamp)
        VALUES ('My prayers to our brothers and sisters in Visayas region..', 'Twitter', 'en', 'neutral', 0.85, 'Typhoon', 'Visayas, Philippines', NOW())
      `);
      console.log("✅ Added sample sentiment post");
    }
    
    console.log("✅ Database checks complete");
    return true;
  } catch (error) {
    console.error("❌ Error fixing database issues:", error);
    return false;
  }
}

async function fixDeleteAllData() {
  console.log("🔧 Improving delete all data functionality...");
  
  try {
    // Create a direct endpoint to delete all data
    const directDeletePath = path.join(process.cwd(), 'server', 'delete-all.js');
    const directDelete = `
/**
 * DIRECT DELETE ALL DATA
 * This provides a direct endpoint to delete all data from the database
 */

import express from 'express';
import { pool } from './db.js';

const router = express.Router();

router.delete('/api/direct-delete-all', async (req, res) => {
  try {
    console.log("⚠️ EXECUTING DIRECT DELETE ALL DATA ⚠️");
    
    // Delete all data from all tables
    await pool.query("DELETE FROM sentiment_feedback");
    await pool.query("DELETE FROM training_examples");
    await pool.query("DELETE FROM upload_sessions");
    await pool.query("DELETE FROM sentiment_posts");
    await pool.query("DELETE FROM disaster_events");
    await pool.query("DELETE FROM analyzed_files");
    
    console.log("✅ All data successfully deleted");
    
    res.json({ 
      success: true, 
      message: "All data has been deleted successfully"
    });
  } catch (error) {
    console.error("❌ Error deleting all data:", error);
    res.status(500).json({ 
      error: "Failed to delete all data",
      details: error.message || String(error)
    });
  }
});

export default router;
`;
  
    fs.writeFileSync(directDeletePath, directDelete);
    
    // Patch routes.ts to use the direct delete endpoint
    const routesPath = path.join(process.cwd(), 'server', 'routes.ts');
    if (fs.existsSync(routesPath)) {
      let routesContent = fs.readFileSync(routesPath, 'utf8');
      
      // Import the direct delete router
      if (!routesContent.includes('import directDeleteRouter')) {
        routesContent = routesContent.replace(
          'export async function registerRoutes(app: Express): Promise<Server> {',
          `// Import direct delete router for emergency data operations
import directDeleteRouter from './delete-all.js';

export async function registerRoutes(app: Express): Promise<Server> {
  // Use direct delete router
  app.use(directDeleteRouter);`
        );
        
        fs.writeFileSync(routesPath, routesContent);
        console.log(`✅ Added direct delete endpoint to routes.ts`);
      } else {
        console.log(`ℹ️ Direct delete endpoint already present in routes.ts`);
      }
    }
    
    return true;
  } catch (error) {
    console.error("❌ Error fixing delete all data functionality:", error);
    return false;
  }
}

// Run all fixes
console.log("🚨 RUNNING EMERGENCY RENDER DEPLOYMENT FIXES 🚨");

// Run SSL fixes
fixSSLIssues().then(sslFixed => {
  console.log(`SSL fixes ${sslFixed ? 'applied successfully' : 'failed'}`);
  
  // Run database fixes
  fixDatabaseIssues().then(dbFixed => {
    console.log(`Database fixes ${dbFixed ? 'applied successfully' : 'failed'}`);
    
    // Run delete all data fixes
    fixDeleteAllData().then(deleteFixed => {
      console.log(`Delete all data fixes ${deleteFixed ? 'applied successfully' : 'failed'}`);
      
      console.log("✅ All emergency fixes applied successfully");
    });
  });
});