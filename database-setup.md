# Database Setup Instructions for Render

> **UPDATE**: We've now updated the Dockerfile and render.yaml to automatically run database migrations on startup. The steps below should no longer be necessary for new deployments after April 2, 2025.

If you still need to run migrations manually, here's how:

1. Go to your Render dashboard
2. Open the PostgreSQL database instance
3. Click on "Connect" and select "External Connection"
4. Use a PostgreSQL client (like psql or pgAdmin) to connect using the provided connection string
5. Run the SQL commands in the `migrations/add_missing_columns.sql` file

SQL to execute:
```sql
-- Add missing columns in Render deployment
ALTER TABLE IF EXISTS sentiment_posts ADD COLUMN IF NOT EXISTS ai_trust_message TEXT;
ALTER TABLE IF EXISTS sentiment_feedback ADD COLUMN IF NOT EXISTS ai_trust_message TEXT;
ALTER TABLE IF EXISTS sentiment_feedback ADD COLUMN IF NOT EXISTS possible_trolling BOOLEAN DEFAULT FALSE;
ALTER TABLE IF EXISTS sentiment_feedback ADD COLUMN IF NOT EXISTS training_error TEXT;

-- Create training_examples table if not exists
CREATE TABLE IF NOT EXISTS training_examples (
  id SERIAL PRIMARY KEY,
  text TEXT NOT NULL UNIQUE,
  text_key TEXT NOT NULL UNIQUE,
  sentiment TEXT NOT NULL,
  language TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.95,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Alternative: Connect using Shell on Render

1. Go to your Render dashboard
2. Open the Web Service (disaster-monitoring-app)
3. Go to the "Shell" tab
4. Connect to the PostgreSQL database using:
```
psql $DATABASE_URL
```
5. Copy and paste the SQL commands above

After running these commands, the database schema will be updated to match the latest version, and the application should work correctly.