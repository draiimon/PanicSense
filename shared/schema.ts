import { pgTable, text, serial, integer, boolean, timestamp, real, json } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const sentimentPosts = pgTable("sentiment_posts", {
  id: serial("id").primaryKey(),
  text: text("text").notNull(),
  timestamp: timestamp("timestamp").notNull().defaultNow(),
  source: text("source"),
  language: text("language"),
  sentiment: text("sentiment").notNull(),
  confidence: real("confidence").notNull(),
  location: text("location"),
  disasterType: text("disaster_type"),
  fileId: integer("file_id"),
  explanation: text("explanation") // Added explanation field
});

export const disasterEvents = pgTable("disaster_events", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  description: text("description"),
  timestamp: timestamp("timestamp").notNull().defaultNow(),
  location: text("location"),
  type: text("type").notNull(),
  sentimentImpact: text("sentiment_impact"),
});

export const analyzedFiles = pgTable("analyzed_files", {
  id: serial("id").primaryKey(),
  originalName: text("original_name").notNull(),
  storedName: text("stored_name").notNull(),
  timestamp: timestamp("timestamp").notNull().defaultNow(),
  recordCount: integer("record_count").notNull(),
  evaluationMetrics: json("evaluation_metrics"),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertSentimentPostSchema = createInsertSchema(sentimentPosts).pick({
  text: true,
  timestamp: true,
  source: true,
  language: true,
  sentiment: true,
  confidence: true,
  location: true,
  disasterType: true,
  fileId: true,
  explanation: true // Added explanation field to the insert schema
});

export const insertDisasterEventSchema = createInsertSchema(disasterEvents).pick({
  name: true,
  description: true,
  timestamp: true,
  location: true,
  type: true,
  sentimentImpact: true,
});

export const insertAnalyzedFileSchema = createInsertSchema(analyzedFiles).pick({
  originalName: true,
  storedName: true,
  recordCount: true,
  evaluationMetrics: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

export type InsertSentimentPost = z.infer<typeof insertSentimentPostSchema>;
export type SentimentPost = typeof sentimentPosts.$inferSelect;

export type InsertDisasterEvent = z.infer<typeof insertDisasterEventSchema>;
export type DisasterEvent = typeof disasterEvents.$inferSelect;

export type InsertAnalyzedFile = z.infer<typeof insertAnalyzedFileSchema>;
export type AnalyzedFile = typeof analyzedFiles.$inferSelect;