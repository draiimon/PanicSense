import pkg from 'pg';
const { Pool } = pkg;
import { drizzle } from 'drizzle-orm/node-postgres';
import * as schema from "@shared/schema";

if (!process.env.DATABASE_URL) {
  throw new Error(
    "DATABASE_URL must be set. Did you forget to provision a database?",
  );
}

// Always use SSL for Neon PostgreSQL database regardless of environment
const isNeonDb = process.env.DATABASE_URL?.includes('neon.tech') || false;
const sslConfig = isNeonDb ? { ssl: { rejectUnauthorized: false } } : {};

export const pool = new Pool({ 
  connectionString: process.env.DATABASE_URL,
  ...sslConfig
});

export const db = drizzle(pool, { schema });

// Log connection success/failure for debugging
pool.on('error', (err) => {
  console.error('Unexpected database error:', err);
});

console.log(`Database connection initialized ${isNeonDb ? 'with SSL for Neon PostgreSQL' : 'without SSL'}`);
