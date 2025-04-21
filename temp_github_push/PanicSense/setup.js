#!/usr/bin/env node

/**
 * First-time setup script for PanicSense PH
 * This script runs automatically the first time a user clones the repository
 * and helps them set up their environment.
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { execSync } = require('child_process');
const crypto = require('crypto');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  blue: '\x1b[34m'
};

// Print welcome message
console.log(`
${colors.bright}${colors.blue}==============================================${colors.reset}
${colors.bright}${colors.blue}       WELCOME TO PANICSENSE PH SETUP        ${colors.reset}
${colors.bright}${colors.blue}==============================================${colors.reset}

${colors.bright}This script will help you set up your development environment.${colors.reset}
`);

// Function to check if a command exists
function commandExists(command) {
  try {
    execSync(`which ${command}`, { stdio: 'ignore' });
    return true;
  } catch (e) {
    return false;
  }
}

// Function to check system requirements
async function checkRequirements() {
  console.log(`${colors.cyan}Checking system requirements...${colors.reset}`);
  
  // Check Node.js
  try {
    const nodeVersion = execSync('node --version').toString().trim();
    console.log(`${colors.green}✓ Node.js ${nodeVersion}${colors.reset}`);
  } catch (e) {
    console.log(`${colors.red}✗ Node.js not found${colors.reset}`);
    console.log(`Please install Node.js 20.x or later: https://nodejs.org/`);
    process.exit(1);
  }
  
  // Check Python
  try {
    const pythonCommand = commandExists('python3') ? 'python3' : 'python';
    const pythonVersion = execSync(`${pythonCommand} --version`).toString().trim();
    console.log(`${colors.green}✓ ${pythonVersion}${colors.reset}`);
  } catch (e) {
    console.log(`${colors.red}✗ Python not found${colors.reset}`);
    console.log(`Please install Python 3.11 or later: https://www.python.org/downloads/`);
    process.exit(1);
  }
  
  // Check PostgreSQL
  try {
    if (commandExists('psql')) {
      const pgVersion = execSync('psql --version').toString().trim();
      console.log(`${colors.green}✓ ${pgVersion}${colors.reset}`);
    } else {
      console.log(`${colors.yellow}⚠ PostgreSQL client not found${colors.reset}`);
      console.log(`We recommend installing PostgreSQL 15.x: https://www.postgresql.org/download/`);
    }
  } catch (e) {
    console.log(`${colors.yellow}⚠ PostgreSQL check failed${colors.reset}`);
    console.log(`We recommend installing PostgreSQL 15.x: https://www.postgresql.org/download/`);
  }
  
  console.log(`${colors.green}System requirements check completed.${colors.reset}`);
  console.log();
}

// Function to set up environment file
async function setupEnvironment() {
  console.log(`${colors.cyan}Setting up environment variables...${colors.reset}`);
  
  // Check if .env file exists
  const envPath = path.join(__dirname, '.env');
  const envExamplePath = path.join(__dirname, '.env.example');
  
  if (fs.existsSync(envPath)) {
    console.log(`${colors.green}✓ .env file already exists${colors.reset}`);
    return;
  }
  
  if (!fs.existsSync(envExamplePath)) {
    console.log(`${colors.red}✗ .env.example file not found${colors.reset}`);
    process.exit(1);
  }
  
  // Read env example
  let envExample = fs.readFileSync(envExamplePath, 'utf8');
  
  // Generate a random session secret
  const sessionSecret = crypto.randomBytes(32).toString('hex');
  envExample = envExample.replace('SESSION_SECRET=replace_with_random_secure_string', `SESSION_SECRET=${sessionSecret}`);
  
  // Ask for database connection details
  const questions = [
    {
      name: 'dbHost',
      message: 'PostgreSQL host',
      default: 'localhost'
    },
    {
      name: 'dbPort',
      message: 'PostgreSQL port',
      default: '5432'
    },
    {
      name: 'dbUser',
      message: 'PostgreSQL username',
      default: 'postgres'
    },
    {
      name: 'dbPassword',
      message: 'PostgreSQL password',
      default: 'postgres'
    },
    {
      name: 'dbName',
      message: 'PostgreSQL database name',
      default: 'panicsense'
    }
  ];
  
  const answers = {};
  
  for (const q of questions) {
    answers[q.name] = await new Promise(resolve => {
      rl.question(`${q.message} (${q.default}): `, (answer) => {
        resolve(answer || q.default);
      });
    });
  }
  
  // Construct database URL
  const dbUrl = `postgres://${answers.dbUser}:${answers.dbPassword}@${answers.dbHost}:${answers.dbPort}/${answers.dbName}`;
  envExample = envExample.replace('DATABASE_URL=postgres://username:password@host:port/database_name', `DATABASE_URL=${dbUrl}`);
  
  // Write to .env file
  fs.writeFileSync(envPath, envExample);
  console.log(`${colors.green}✓ .env file created successfully${colors.reset}`);
  console.log();
}

// Function to install dependencies
async function installDependencies() {
  console.log(`${colors.cyan}Installing Node.js dependencies...${colors.reset}`);
  
  try {
    execSync('npm install', { stdio: 'inherit' });
    console.log(`${colors.green}✓ Node.js dependencies installed successfully${colors.reset}`);
  } catch (e) {
    console.log(`${colors.red}✗ Failed to install Node.js dependencies${colors.reset}`);
    console.error(e);
    process.exit(1);
  }
  
  console.log();
}

// Function to set up database
async function setupDatabase() {
  console.log(`${colors.cyan}Setting up database...${colors.reset}`);
  
  try {
    execSync('npm run db:push', { stdio: 'inherit' });
    console.log(`${colors.green}✓ Database setup completed successfully${colors.reset}`);
  } catch (e) {
    console.log(`${colors.red}✗ Failed to set up database${colors.reset}`);
    console.log(`Please make sure your PostgreSQL server is running and credentials are correct.`);
    console.error(e);
  }
  
  console.log();
}

// Main function
async function main() {
  try {
    await checkRequirements();
    await setupEnvironment();
    const response = await new Promise(resolve => {
      rl.question(`${colors.yellow}Do you want to install dependencies now? (Y/n): ${colors.reset}`, (answer) => {
        resolve(answer.toLowerCase() !== 'n');
      });
    });
    
    if (response) {
      await installDependencies();
      
      const setupDb = await new Promise(resolve => {
        rl.question(`${colors.yellow}Do you want to set up the database now? (Y/n): ${colors.reset}`, (answer) => {
          resolve(answer.toLowerCase() !== 'n');
        });
      });
      
      if (setupDb) {
        await setupDatabase();
      }
    }
    
    console.log(`
${colors.bright}${colors.green}==============================================${colors.reset}
${colors.bright}${colors.green}       SETUP COMPLETED SUCCESSFULLY          ${colors.reset}
${colors.bright}${colors.green}==============================================${colors.reset}

${colors.bright}You can now start the application:${colors.reset}

  ${colors.cyan}npm run dev${colors.reset}

${colors.bright}Thank you for installing PanicSense PH!${colors.reset}
`);
  } catch (error) {
    console.error(`${colors.red}Error during setup:${colors.reset}`, error);
  } finally {
    rl.close();
  }
}

// Run the main function
main();