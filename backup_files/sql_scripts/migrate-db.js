import { exec as execCallback } from 'child_process';
import { promisify } from 'util';
import postgres from 'postgres';

const exec = promisify(execCallback);

async function runMigration() {
  console.log('Starting database migration...');
  
  try {
    // First, try using drizzle-kit push with the force flag
    console.log('Attempting to push schema with force flag...');
    try {
      const { stdout, stderr } = await exec('npx drizzle-kit push --force');
      console.log(stdout);
      if (stderr) console.error(stderr);
    } catch (error) {
      console.error(`Error running drizzle-kit push: ${error.message}`);
      throw error;
    }
    
    console.log('Schema push completed successfully');
    
    // Check database connection
    console.log('Verifying database connection...');
    const connectionString = process.env.DATABASE_URL;
    const client = postgres(connectionString, { max: 1 });
    
    try {
      const result = await client`SELECT current_database()`;
      console.log(`Connected successfully to database: ${result[0].current_database}`);
    } catch (err) {
      console.error('Database connection error:', err);
      throw err;
    } finally {
      await client.end();
    }
    
    console.log('Migration completed successfully!');
    
  } catch (error) {
    console.error('Migration failed:', error);
    process.exit(1);
  }
}

runMigration();