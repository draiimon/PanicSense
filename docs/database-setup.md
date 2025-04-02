# Database Setup Instructions for Render

> **IMPORTANT UPDATE**: A new comprehensive database migration script has been added. Deployments after April 2, 2025 should automatically include all required tables and columns. If you're experiencing database issues, use one of the methods below.

## Option 1: Using the Quick Fix Script (Recommended)

This is the easiest method for fixing database issues in an existing deployment:

1. Go to your Render dashboard
2. Open the Web Service (disaster-monitoring-app)
3. Go to the "Shell" tab
4. Run the following command:
   ```
   bash quick-fix.sh
   ```
5. Restart your service from the Render dashboard

## Option 2: Manual Database Setup

If the quick fix script doesn't work, you can manually run the SQL commands:

1. Go to your Render dashboard
2. Open the Web Service (disaster-monitoring-app)
3. Go to the "Shell" tab
4. Connect to the PostgreSQL database using:
   ```
   psql $DATABASE_URL
   ```
5. Copy and paste the SQL commands from `migrations/complete_schema.sql` file

## Option 3: External Database Connection

If you prefer using your own database client:

1. Go to your Render dashboard
2. Open the PostgreSQL database instance
3. Click on "Connect" and select "External Connection"
4. Use a PostgreSQL client (like psql or pgAdmin) to connect using the provided connection string
5. Run the SQL commands from `migrations/complete_schema.sql` file

After running these commands, the database schema will be updated to match the latest version, and the application should work correctly.