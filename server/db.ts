import { Pool, neonConfig } from '@neondatabase/serverless';
import { drizzle } from 'drizzle-orm/neon-serverless';
import ws from "ws";
import * as schema from "@shared/schema";

// Enable detailed logging for Render deployment
const isDevelopment = process.env.NODE_ENV === 'development';
const isRender = !!process.env.RENDER;

if (isDevelopment || isRender) {
  console.log('========================================');
  console.log('DATABASE CONNECTION DETAILS (DEVELOPMENT MODE)');
  console.log(`Environment: ${process.env.NODE_ENV}`);
  console.log(`Running on Render: ${isRender ? 'Yes' : 'No'}`);
  console.log(`Database URL configured: ${!!process.env.DATABASE_URL}`);
  console.log(`Neon Database URL configured: ${!!process.env.NEON_DATABASE_URL}`);
  console.log('========================================');
}

// Configure Neon database to use WebSockets
neonConfig.webSocketConstructor = ws;

// Prioritize Neon database URL if available, fall back to regular DATABASE_URL
let databaseUrl = process.env.NEON_DATABASE_URL || process.env.DATABASE_URL;

if (!databaseUrl) {
  console.error('NO DATABASE URL FOUND!');
  console.error('Environment variables available:', Object.keys(process.env).join(', '));
  throw new Error(
    "DATABASE_URL or NEON_DATABASE_URL must be set. Did you forget to provision a database?",
  );
}

// Remove the 'DATABASE_URL=' prefix if it exists (happens on some platforms)
if (databaseUrl.startsWith('DATABASE_URL=')) {
  databaseUrl = databaseUrl.substring('DATABASE_URL='.length);
}

if (isDevelopment || isRender) {
  // Show connection type for debugging
  console.log(`🔌 Using database connection type: ${databaseUrl.split(':')[0]}`);
  console.log(`🔒 SSL Required: ${process.env.DB_SSL_REQUIRED || 'true'}`);
}

// Connect to the database
export const pool = new Pool({ 
  connectionString: databaseUrl,
  // Always use SSL with Neon, but don't reject unauthorized for flexibility in development
  ssl: { rejectUnauthorized: false }
});

// Debug logging
if (isDevelopment || isRender) {
  console.log('🔄 Creating database connection with schema tables:');
  console.log(Object.keys(schema).filter(key => key !== 'default').join(', '));
}

// Create the drizzle ORM instance
export const db = drizzle(pool, { schema });

// Export a function to test the database connection
export async function testDatabaseConnection(): Promise<boolean> {
  try {
    console.log('🔄 Testing database connection...');
    const client = await pool.connect();
    const result = await client.query('SELECT NOW() as now');
    console.log(`✅ Database connection successful! Server time: ${result.rows[0].now}`);
    client.release();
    return true;
  } catch (error) {
    console.error('❌ DATABASE CONNECTION FAILED:', error.message);
    console.error('Stack trace:', error.stack);
    return false;
  }
}
