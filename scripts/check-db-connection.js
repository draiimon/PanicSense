#!/usr/bin/env node

/**
 * Database Connection Checker for PanicSense
 * Author: Mark Andrei R. Castillo
 * 
 * This script tests the connection to the Neon database
 * and displays basic information about the database.
 */

import 'dotenv/config';
import pg from 'pg';
const { Pool } = pg;

async function checkConnection() {
  console.log('🔍 PanicSense Database Connection Checker');
  console.log('==========================================');
  
  // Check if DATABASE_URL is set
  if (!process.env.DATABASE_URL) {
    console.error('❌ ERROR: DATABASE_URL environment variable is not set!');
    console.log('Please check your .env file or environment variables.');
    process.exit(1);
  }
  
  // Check if it's a Neon database
  const isNeonDb = process.env.DATABASE_URL.includes('neon.tech');
  console.log(`Database Type: ${isNeonDb ? '☁️ Neon PostgreSQL (Cloud)' : '🖥️ Standard PostgreSQL'}`);
  
  // Always use SSL for Neon Database regardless of environment
  const sslConfig = isNeonDb ? { ssl: { rejectUnauthorized: false } } : false;
  
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: sslConfig,
    connectionTimeoutMillis: 10000,
  });
  
  try {
    console.log('Connecting to database...');
    
    // Test basic connection
    const result = await pool.query('SELECT NOW() as current_time');
    console.log(`✅ Connection successful! Server time: ${result.rows[0].current_time}`);
    
    // Get database info
    const dbInfo = await pool.query(`
      SELECT current_database() as db_name, 
             current_user as db_user,
             version() as db_version
    `);
    
    console.log('\n📊 Database Information:');
    console.log(`Database Name: ${dbInfo.rows[0].db_name}`);
    console.log(`Connected User: ${dbInfo.rows[0].db_user}`);
    console.log(`Version: ${dbInfo.rows[0].db_version.split(',')[0]}`);
    
    // Check tables
    const tableCount = await pool.query(`
      SELECT COUNT(*) as table_count
      FROM information_schema.tables
      WHERE table_schema = 'public'
    `);
    
    console.log(`\n📋 Database contains ${tableCount.rows[0].table_count} tables`);
    
    if (parseInt(tableCount.rows[0].table_count) > 0) {
      const tables = await pool.query(`
        SELECT table_name, 
               (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
        FROM information_schema.tables t
        WHERE table_schema = 'public'
        ORDER BY table_name
      `);
      
      console.log('\nAvailable tables:');
      tables.rows.forEach(table => {
        console.log(`- ${table.table_name} (${table.column_count} columns)`);
      });
    }
    
    console.log('\n✅ Database connection check completed successfully!');
    
  } catch (error) {
    console.error('❌ Connection failed!', error.message);
    console.error('Details:', error);
    process.exit(1);
  } finally {
    await pool.end();
  }
}

checkConnection().catch(err => {
  console.error('Unhandled error:', err);
  process.exit(1);
});