import pkg from 'pg';
const { Pool } = pkg;
import { drizzle } from 'drizzle-orm/node-postgres';
import * as schema from "@shared/schema";
import { neon, neonConfig } from '@neondatabase/serverless';

if (!process.env.DATABASE_URL) {
  throw new Error(
    "DATABASE_URL must be set. Did you forget to provision a database?",
  );
}

// Configure Neon for better performance
neonConfig.fetchConnectionCache = true;

// Create a direct Neon connection - this is the most reliable for Neon Cloud
export const neonConnection = neon(process.env.DATABASE_URL);

// Create a pool for compatibility with existing code
export const pool = new Pool({ 
  connectionString: process.env.DATABASE_URL,
  max: 5, // reduced max connections
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
  ssl: {
    rejectUnauthorized: false // For Neon cloud
  }
});

// Handle pool errors to prevent the app from crashing
pool.on('error', (err) => {
  console.error('Unexpected error on idle database client', err);
  // Don't crash on connection errors
});

// Export both drizzle instances
export const db = drizzle(pool, { schema });

// Method to test database connection
export async function testDatabaseConnection() {
  try {
    const result = await pool.query('SELECT NOW()');
    return {
      connected: true,
      timestamp: result.rows[0].now
    };
  } catch (error) {
    console.error('Database connection test failed:', error);
    return {
      connected: false,
      error: error instanceof Error ? error.message : String(error)
    };
  }
}
