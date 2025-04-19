/**
 * Simple database fix function.
 * Directly creates tables with all the necessary columns.
 * @returns Promise<boolean> True if the fix was applied successfully
 */
import { db } from './db';
import * as schema from '../shared/schema';
import { sql } from 'drizzle-orm';

export async function simpleDbFix() {
  console.log("Running simple database fix...");
  
  try {
    // Create all tables from our schema
    const tableCreatePromises = [
      // Users table
      db.execute(sql`
        CREATE TABLE IF NOT EXISTS "users" (
          "id" serial PRIMARY KEY,
          "username" varchar(255) NOT NULL UNIQUE,
          "password" varchar(255) NOT NULL,
          "email" varchar(255),
          "full_name" varchar(255),
          "role" varchar(50) DEFAULT 'user',
          "created_at" timestamp DEFAULT now()
        )
      `),
      
      // Sessions table
      db.execute(sql`
        CREATE TABLE IF NOT EXISTS "sessions" (
          "id" serial PRIMARY KEY,
          "user_id" integer REFERENCES "users"("id") ON DELETE CASCADE,
          "token" varchar(255) NOT NULL UNIQUE,
          "created_at" timestamp DEFAULT now(),
          "expires_at" timestamp
        )
      `),
      
      // Sentiment posts table
      db.execute(sql`
        CREATE TABLE IF NOT EXISTS "sentiment_posts" (
          "id" serial PRIMARY KEY,
          "text" text NOT NULL,
          "timestamp" timestamp DEFAULT now(),
          "source" varchar(255),
          "language" varchar(50),
          "sentiment" varchar(50),
          "confidence" double precision DEFAULT 0,
          "file_id" integer,
          "disaster_type" varchar(255),
          "location" varchar(255)
        )
      `),
      
      // Disaster events table
      db.execute(sql`
        CREATE TABLE IF NOT EXISTS "disaster_events" (
          "id" serial PRIMARY KEY,
          "name" varchar(255) NOT NULL,
          "type" varchar(50),
          "start_date" timestamp,
          "end_date" timestamp,
          "location" varchar(255),
          "description" text,
          "impact_level" varchar(50),
          "created_at" timestamp DEFAULT now()
        )
      `),
      
      // Analyzed files table
      db.execute(sql`
        CREATE TABLE IF NOT EXISTS "analyzed_files" (
          "id" serial PRIMARY KEY,
          "filename" varchar(255) NOT NULL,
          "original_filename" varchar(255),
          "file_size" integer,
          "record_count" integer,
          "processed_count" integer,
          "success_count" integer,
          "error_count" integer,
          "source_type" varchar(50),
          "upload_date" timestamp DEFAULT now(),
          "processing_time" integer,
          "status" varchar(50),
          "user_id" integer,
          "metrics" jsonb
        )
      `),
      
      // Sentiment feedback table
      db.execute(sql`
        CREATE TABLE IF NOT EXISTS "sentiment_feedback" (
          "id" serial PRIMARY KEY,
          "text" text NOT NULL,
          "original_sentiment" varchar(50),
          "corrected_sentiment" varchar(50),
          "submitted_at" timestamp DEFAULT now(),
          "user_id" integer,
          "trained" boolean DEFAULT false
        )
      `),
      
      // Training examples table
      db.execute(sql`
        CREATE TABLE IF NOT EXISTS "training_examples" (
          "id" serial PRIMARY KEY,
          "text" text NOT NULL,
          "sentiment" varchar(50) NOT NULL,
          "language" varchar(50),
          "created_at" timestamp DEFAULT now(),
          "source" varchar(255)
        )
      `),
      
      // Upload sessions table
      db.execute(sql`
        CREATE TABLE IF NOT EXISTS "upload_sessions" (
          "id" serial PRIMARY KEY,
          "session_id" varchar(255) NOT NULL UNIQUE,
          "status" varchar(50) NOT NULL,
          "progress" jsonb,
          "created_at" timestamp DEFAULT now(),
          "updated_at" timestamp DEFAULT now(),
          "user_id" integer
        )
      `),
      
      // Profile images table
      db.execute(sql`
        CREATE TABLE IF NOT EXISTS "profile_images" (
          "id" serial PRIMARY KEY,
          "filename" varchar(255) NOT NULL,
          "original_filename" varchar(255),
          "file_size" integer,
          "upload_date" timestamp DEFAULT now(),
          "user_id" integer
        )
      `)
    ];
    
    await Promise.all(tableCreatePromises);
    console.log("All tables created successfully");
    
    return true;
  } catch (error) {
    console.error("Error creating tables:", error);
    return false;
  }
}