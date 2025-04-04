import pkg from 'pg';
const { Pool } = pkg;
import { drizzle } from 'drizzle-orm/node-postgres';
import * as schema from "@shared/schema";

if (!process.env.DATABASE_URL) {
  throw new Error(
    "DATABASE_URL must be set. Did you forget to provision a database?",
  );
}

// Enhanced pool with better error handling and automatic reconnection
export const pool = new Pool({ 
  connectionString: process.env.DATABASE_URL,
  max: 20, // Maximum number of clients in the pool
  idleTimeoutMillis: 30000, // Time a client can be idle before being closed
  connectionTimeoutMillis: 5000, // Return an error after 5 seconds if connection not established
  maxUses: 7500 // Close and replace a connection after a client has used it 7500 times (prevents memory leaks)
});

// Log connection events
pool.on('connect', (client) => {
  console.log('New database connection established');
});

pool.on('error', (err, client) => {
  console.error('Unexpected error on idle database client', err);
});

export const db = drizzle(pool, { schema });

// Utility function for retrying database operations
export async function withRetry<T>(operation: () => Promise<T>, retries = 3, delay = 1000): Promise<T> {
  let lastError: any;
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await operation();
    } catch (error: any) {
      lastError = error;
      
      // If it's a connection error that might be temporary, retry
      if (error.code === '57P01' || error.code === '08006' || error.code === '08001' || error.code === '08004' || error.code === '57P02') {
        console.warn(`Database operation failed (attempt ${attempt}/${retries}): ${error.message}. Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
        // Increase delay for next attempt (exponential backoff)
        delay *= 2;
        continue;
      }
      
      // For other errors, don't retry
      throw error;
    }
  }
  
  // If we've exhausted all retries
  throw lastError;
}
