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

export const pool = new Pool({ 
  connectionString: process.env.DATABASE_URL,
  ...sslConfig
});

export const db = drizzle(pool, { schema });

// Log connection success/failure for debugging
pool.on('error', (err) => {
  console.error('Unexpected database error:', err);
});

console.log(`Database connection initialized with${isProduction ? '' : 'out'} SSL`);
