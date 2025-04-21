/**
 * Simple ES Module compatible database connection
 * This is a direct fallback file for Render.com deployment
 */

import pg from 'pg';
const { Pool } = pg;

// Prioritize Neon database URL if available, fall back to regular DATABASE_URL
let databaseUrl = process.env.NEON_DATABASE_URL || process.env.DATABASE_URL;

if (!databaseUrl) {
  console.error("⚠️ WARNING: No DATABASE_URL or NEON_DATABASE_URL environment variable found");
  databaseUrl = ""; // Empty string will cause connect error later, but prevent crash here
}

// Remove the 'DATABASE_URL=' prefix if it exists (sometimes happens in environment vars)
if (databaseUrl.startsWith('DATABASE_URL=')) {
  databaseUrl = databaseUrl.substring('DATABASE_URL='.length);
}

// Log database type but not the connection string (for security)
console.log(`Using database connection type: ${databaseUrl.split(':')[0]}`);

// Create the pool with SSL enabled (important for Neon.tech and Render.com PostgreSQL)
export const pool = new Pool({ 
  connectionString: databaseUrl,
  ssl: { rejectUnauthorized: false } // Always use SSL 
});

// Export in a way compatible with older requires
export default {
  pool
};