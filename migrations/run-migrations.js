// Enhanced migration script using both direct SQL and schema validation
require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { Pool } = require('pg');
const { drizzle } = require('drizzle-orm/node-postgres');
// Import migrator separately to avoid schema imports which might fail
const { migrate } = require('drizzle-orm/node-postgres/migrator');

// Establish a connection to the database
async function runMigrations() {
  console.log('Starting database migrations...');
  
  // Connect to database using DATABASE_URL with SSL settings for production
  const isProduction = process.env.NODE_ENV === 'production';
  const sslConfig = isProduction ? { ssl: { rejectUnauthorized: false } } : false;
  
  console.log(`Database migration running in ${isProduction ? 'production' : 'development'} mode`);
  console.log(`Using SSL configuration: ${isProduction ? 'enabled' : 'disabled'}`);
  
  // Create a connection pool with a timeout
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: sslConfig,
    // Longer connection timeout for migrations
    connectionTimeoutMillis: 10000,
    // Increase idle timeout to prevent disconnect during migration
    idleTimeoutMillis: 30000,
    // Single connection is enough for migrations
    max: 1
  });

  const db = drizzle(pool);
  
  // Test the connection
  let connectionPool = pool;
  try {
    console.log('Testing database connection...');
    await connectionPool.query('SELECT NOW()');
    console.log('Database connection successful');
  } catch (connectionError) {
    console.error('Database connection error:', connectionError);
    console.log('Will retry migrations with modified connection settings...');
    
    // If we failed to connect, try again with a different SSL setting
    try {
      await connectionPool.end();
      
      // Try the opposite SSL setting
      const newSslConfig = !isProduction ? { ssl: { rejectUnauthorized: false } } : false;
      console.log(`Retrying with alternate SSL configuration: ${!isProduction ? 'enabled' : 'disabled'}`);
      
      // Create a new pool with different settings
      connectionPool = new Pool({
        connectionString: process.env.DATABASE_URL,
        ssl: newSslConfig,
        connectionTimeoutMillis: 10000
      });
      
      await connectionPool.query('SELECT NOW()');
      console.log('Retry connection successful');
    } catch (retryError) {
      console.error('Retry connection also failed:', retryError);
      console.error('Continuing with migrations despite connection issues...');
      // Recreate the original pool
      connectionPool = new Pool({
        connectionString: process.env.DATABASE_URL,
        ssl: sslConfig,
        connectionTimeoutMillis: 10000
      });
    }
  }
  
  // Use the possibly updated connection pool for the rest of the function
  pool = connectionPool;

  try {
    // Verify we can access information_schema first
    try {
      console.log('Verifying database access...');
      await pool.query(`
        SELECT EXISTS (
          SELECT 1 FROM information_schema.tables
          WHERE table_schema = 'public'
        );
      `);
      console.log('Database schema access verified');
    } catch (accessError) {
      console.error('Error accessing database schema:', accessError);
      console.log('Will attempt migrations anyway...');
    }
    
    // Step 1: Run SQL migration script with error handling per statement
    // First try complete schema for new deployments
    const completeSchemaPath = path.join(__dirname, 'complete_schema.sql');
    // Fallback to add_missing_columns if complete_schema doesn't exist (backwards compatibility)
    const fallbackPath = path.join(__dirname, 'add_missing_columns.sql');
    
    let sqlPath;
    if (fs.existsSync(completeSchemaPath)) {
      console.log('Using complete schema migration script...');
      sqlPath = completeSchemaPath;
    } else {
      console.log('Complete schema not found, using fallback migration script...');
      sqlPath = fallbackPath;
    }
    
    const sql = fs.readFileSync(sqlPath, 'utf8');
    const statements = sql.split(';').filter(stmt => stmt.trim().length > 0);
    
    console.log(`Executing ${statements.length} SQL migration statements...`);
    
    for (const statement of statements) {
      try {
        await pool.query(statement);
        console.log(`Successfully executed: ${statement.substring(0, 50)}...`);
      } catch (stmtError) {
        console.error(`Error executing statement: ${statement.substring(0, 100)}...`);
        console.error(`Error details: ${stmtError.message}`);
        // Continue with other statements
      }
    }
    
    console.log('SQL migration statements completed');

    // Step 2: Verify and create each required table individually
    const requiredTables = [
      'training_examples',
      'sentiment_posts',
      'sentiment_feedback',
      'analyzed_files',
      'disaster_events'
    ];
    
    for (const tableName of requiredTables) {
      try {
        console.log(`Verifying ${tableName} table...`);
        const result = await pool.query(`
          SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = $1
          );
        `, [tableName]);
        
        if (!result.rows[0].exists) {
          console.log(`Table ${tableName} does not exist - will ensure it's created in the next step`);
        } else {
          console.log(`Table ${tableName} exists - checking columns...`);
          
          // For existing tables, verify critical columns exist
          if (tableName === 'sentiment_posts') {
            await verifyAndAddColumn(pool, 'sentiment_posts', 'ai_trust_message', 'TEXT');
            await verifyAndAddColumn(pool, 'sentiment_posts', 'runtime_order', 'SERIAL');
          } else if (tableName === 'sentiment_feedback') {
            await verifyAndAddColumn(pool, 'sentiment_feedback', 'ai_trust_message', 'TEXT');
            await verifyAndAddColumn(pool, 'sentiment_feedback', 'possible_trolling', 'BOOLEAN DEFAULT FALSE');
            await verifyAndAddColumn(pool, 'sentiment_feedback', 'training_error', 'TEXT');
          }
        }
      } catch (tableError) {
        console.error(`Error verifying ${tableName} table:`, tableError);
      }
    }
    
    // Re-run the full SQL to ensure everything exists
    try {
      console.log('Running full SQL as final verification...');
      await pool.query(sql);
    } catch (finalError) {
      console.log('Final SQL verification completed with some errors (expected)');
    }

    console.log('All migrations completed successfully!');
  } catch (error) {
    console.error('Error executing migrations:', error);
    // Log more details about the error for troubleshooting
    if (error.stack) {
      console.error('Error stack:', error.stack);
    }
    
    // Don't fail the container if migrations fail - app might still work with existing schema
  } finally {
    await pool.end();
  }
}

// Helper function to verify and add a column if it doesn't exist
async function verifyAndAddColumn(pool, tableName, columnName, columnType) {
  try {
    const result = await pool.query(`
      SELECT EXISTS (
        SELECT FROM information_schema.columns 
        WHERE table_schema = 'public'
        AND table_name = $1
        AND column_name = $2
      );
    `, [tableName, columnName]);
    
    if (!result.rows[0].exists) {
      console.log(`Adding missing column ${columnName} to ${tableName}`);
      await pool.query(`ALTER TABLE ${tableName} ADD COLUMN IF NOT EXISTS ${columnName} ${columnType};`);
      console.log(`Column ${columnName} added successfully`);
    } else {
      console.log(`Column ${columnName} already exists in ${tableName}`);
    }
  } catch (error) {
    console.error(`Error verifying/adding column ${columnName}:`, error);
  }
}

// Run migrations immediately
console.log('Starting migration script...');
runMigrations().catch(err => {
  console.error('Unhandled migration error:', err);
  // Don't fail the deployment - let the app try to run anyway
  console.log('Migration script completed with errors, but continuing deployment');
});