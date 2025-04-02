import pkg from 'pg';
const { Pool } = pkg;
import { drizzle } from 'drizzle-orm/node-postgres';
import * as schema from "@shared/schema";

if (!process.env.DATABASE_URL) {
  throw new Error(
    "DATABASE_URL must be set. Did you forget to provision a database?",
  );
}

const isProduction = process.env.NODE_ENV === 'production';
const sslConfig = isProduction ? { ssl: { rejectUnauthorized: false } } : {};

// Optimized connection pool with better defaults for performance
export const pool = new Pool({ 
  connectionString: process.env.DATABASE_URL,
  max: 20, // Maximum number of clients in the pool (increase from default 10)
  idleTimeoutMillis: 30000, // How long a client is allowed to remain idle (30 seconds)
  connectionTimeoutMillis: 2000, // Return an error after 2 seconds if connection cannot be established
  ...sslConfig
});

export const db = drizzle(pool, { schema });

// Connection event handling for better diagnostics
pool.on('connect', () => {
  console.log('New client connected to PostgreSQL pool');
});

pool.on('remove', () => {
  console.log('Client removed from PostgreSQL pool');
});

pool.on('error', (err) => {
  console.error('Unexpected database error:', err);
});

console.log(`Database connection initialized with${isProduction ? '' : 'out'} SSL - Pool size: 20`);
