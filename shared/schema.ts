import { pgTable, text, serial, integer, boolean, timestamp, real, json, date, pgEnum, relations } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Profile Images Table
export const profileImages = pgTable("profile_images", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  role: text("role").notNull(),
  imageUrl: text("image_url").notNull(),
  description: text("description"),
  createdAt: timestamp("created_at").defaultNow().notNull()
});

export const insertProfileImageSchema = createInsertSchema(profileImages);
export type ProfileImage = typeof profileImages.$inferSelect;
export type InsertProfileImage = z.infer<typeof insertProfileImageSchema>;

// Authentication Tables
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
  email: text("email").notNull().unique(),
  fullName: text("full_name").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  role: text("role").default("user").notNull(),
});

export const sessions = pgTable("sessions", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull().references(() => users.id),
  token: text("token").notNull().unique(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  expiresAt: timestamp("expires_at").notNull(),
});

// Sentiment Analysis Tables
export const sentimentPosts = pgTable("sentiment_posts", {
  id: serial("id").primaryKey(),
  text: text("text").notNull(),
  timestamp: timestamp("timestamp").notNull().defaultNow(),
  source: text("source"),
  language: text("language"),
  sentiment: text("sentiment").notNull(),
  confidence: real("confidence").notNull(),
  location: text("location"), // Now only populated when location is mentioned in text
  disasterType: text("disaster_type"), // Now only populated when disaster type is mentioned
  fileId: integer("file_id"),
  explanation: text("explanation"),
  processedBy: integer("processed_by").references(() => users.id),
  aiTrustMessage: text("ai_trust_message"), // Added for validation messages
});

export const disasterEvents = pgTable("disaster_events", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  description: text("description"),
  timestamp: timestamp("timestamp").notNull().defaultNow(),
  location: text("location"),
  type: text("type").notNull(),
  sentimentImpact: text("sentiment_impact"),
  createdBy: integer("created_by").references(() => users.id),
});

export const analyzedFiles = pgTable("analyzed_files", {
  id: serial("id").primaryKey(),
  originalName: text("original_name").notNull(),
  storedName: text("stored_name").notNull(),
  timestamp: timestamp("timestamp").notNull().defaultNow(),
  recordCount: integer("record_count").notNull(),
  evaluationMetrics: json("evaluation_metrics"),
  uploadedBy: integer("uploaded_by").references(() => users.id),
});

// Feedback data for model training
export const sentimentFeedback = pgTable("sentiment_feedback", {
  id: serial("id").primaryKey(),
  originalPostId: integer("original_post_id").references(() => sentimentPosts.id, { onDelete: "cascade" }),
  originalText: text("original_text").notNull(),
  originalSentiment: text("original_sentiment").notNull(),
  correctedSentiment: text("corrected_sentiment").default(''),
  correctedLocation: text("corrected_location"),
  correctedDisasterType: text("corrected_disaster_type"),
  trainedOn: boolean("trained_on").default(false),
  createdAt: timestamp("created_at").defaultNow(),
  userId: integer("user_id").references(() => users.id),
  aiTrustMessage: text("ai_trust_message"), // Added for validation messages
  possibleTrolling: boolean("possible_trolling").default(false) // Added for validation flags
});

// Training examples for real-time model learning
export const trainingExamples = pgTable("training_examples", {
  id: serial("id").primaryKey(),
  text: text("text").notNull().unique(),
  textKey: text("text_key").notNull().unique(), // Normalized text for matching
  sentiment: text("sentiment").notNull(),
  language: text("language").notNull(),
  confidence: real("confidence").notNull().default(0.95),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Upload sessions for persistence across page refreshes
export const uploadSessions = pgTable("upload_sessions", {
  id: serial("id").primaryKey(),
  sessionId: text("session_id").notNull().unique(),
  status: text("status").notNull().default('active'), // active, complete, canceled, error
  fileId: integer("file_id").references(() => analyzedFiles.id),
  progress: json("progress"), // Store upload progress data
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
  userId: integer("user_id").references(() => users.id),
  serverStartTimestamp: text("server_start_timestamp"), // Store server start timestamp to detect restarts (stored as string)
});

export const insertSentimentFeedbackSchema = createInsertSchema(sentimentFeedback).pick({
  originalPostId: true,
  originalText: true,
  originalSentiment: true,
  correctedSentiment: true,
  correctedLocation: true,
  correctedDisasterType: true,
  userId: true,
  aiTrustMessage: true,
  possibleTrolling: true
});

export const insertTrainingExampleSchema = createInsertSchema(trainingExamples).pick({
  text: true,
  textKey: true,
  sentiment: true,
  language: true,
  confidence: true
});

export const insertUploadSessionSchema = createInsertSchema(uploadSessions).pick({
  sessionId: true,
  status: true,
  fileId: true,
  progress: true,
  userId: true,
  serverStartTimestamp: true
});

// Community Challenge System
export const challengeDifficultyEnum = pgEnum('challenge_difficulty', ['beginner', 'intermediate', 'advanced', 'expert']);
export const challengeStatusEnum = pgEnum('challenge_status', ['draft', 'active', 'completed', 'archived']);
export const userProgressEnum = pgEnum('user_progress_status', ['not_started', 'in_progress', 'completed', 'verified']);

// Challenges Table
export const challenges = pgTable("challenges", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  description: text("description").notNull(),
  instructions: text("instructions").notNull(),
  difficulty: challengeDifficultyEnum("difficulty").notNull().default('beginner'),
  disasterType: text("disaster_type").notNull(),
  imageUrl: text("image_url"),
  pointsValue: integer("points_value").notNull().default(10),
  learningOutcomes: text("learning_outcomes").notNull(),
  status: challengeStatusEnum("status").notNull().default('draft'),
  startDate: date("start_date").notNull(),
  endDate: date("end_date").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
  createdBy: integer("created_by").references(() => users.id),
});

// Challenge Resources
export const challengeResources = pgTable("challenge_resources", {
  id: serial("id").primaryKey(),
  challengeId: integer("challenge_id").notNull().references(() => challenges.id, { onDelete: "cascade" }),
  title: text("title").notNull(),
  resourceType: text("resource_type").notNull(), // video, article, pdf, etc.
  url: text("url").notNull(),
  description: text("description"),
  createdAt: timestamp("created_at").defaultNow(),
});

// User Challenge Progress
export const userChallengeProgress = pgTable("user_challenge_progress", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull().references(() => users.id),
  challengeId: integer("challenge_id").notNull().references(() => challenges.id),
  status: userProgressEnum("status").notNull().default('not_started'),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  verifiedAt: timestamp("verified_at"),
  pointsEarned: integer("points_earned").default(0),
  submissionNotes: text("submission_notes"),
  submissionUrl: text("submission_url"),
  feedbackNotes: text("feedback_notes"),
});

// User Points Leaderboard
export const userPoints = pgTable("user_points", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").notNull().references(() => users.id),
  totalPoints: integer("total_points").notNull().default(0),
  weeklyPoints: integer("weekly_points").notNull().default(0),
  monthlyPoints: integer("monthly_points").notNull().default(0),
  lastUpdated: timestamp("last_updated").defaultNow(),
  badgesEarned: integer("badges_earned").default(0),
  challengesCompleted: integer("challenges_completed").default(0),
});

// Challenge Community Discussions
export const challengeDiscussions = pgTable("challenge_discussions", {
  id: serial("id").primaryKey(),
  challengeId: integer("challenge_id").notNull().references(() => challenges.id),
  userId: integer("user_id").notNull().references(() => users.id),
  content: text("content").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
  parentId: integer("parent_id"), // Will be updated with a relation after table creation
  pinned: boolean("pinned").default(false),
});

// Schema Validation
export const insertUserSchema = createInsertSchema(users).extend({
  confirmPassword: z.string(),
  email: z.string().email("Invalid email format"),
  username: z.string().min(3, "Username must be at least 3 characters"),
  password: z.string().min(8, "Password must be at least 8 characters"),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

export const loginSchema = z.object({
  username: z.string(),
  password: z.string(),
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
  explanation: true,
  processedBy: true,
  aiTrustMessage: true,
});

export const insertDisasterEventSchema = createInsertSchema(disasterEvents).pick({
  name: true,
  description: true,
  timestamp: true,
  location: true,
  type: true,
  sentimentImpact: true,
  createdBy: true,
});

export const insertAnalyzedFileSchema = createInsertSchema(analyzedFiles).pick({
  originalName: true,
  storedName: true,
  recordCount: true,
  evaluationMetrics: true,
  uploadedBy: true,
});

// Type Exports
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type LoginUser = z.infer<typeof loginSchema>;

export type InsertSentimentPost = z.infer<typeof insertSentimentPostSchema>;
export type SentimentPost = typeof sentimentPosts.$inferSelect;

export type InsertDisasterEvent = z.infer<typeof insertDisasterEventSchema>;
export type DisasterEvent = typeof disasterEvents.$inferSelect;

export type InsertAnalyzedFile = z.infer<typeof insertAnalyzedFileSchema>;
export type AnalyzedFile = typeof analyzedFiles.$inferSelect;

export type InsertSentimentFeedback = z.infer<typeof insertSentimentFeedbackSchema>;
export type SentimentFeedback = typeof sentimentFeedback.$inferSelect;

export type InsertTrainingExample = z.infer<typeof insertTrainingExampleSchema>;
export type TrainingExample = typeof trainingExamples.$inferSelect;

export type InsertUploadSession = z.infer<typeof insertUploadSessionSchema>;
export type UploadSession = typeof uploadSessions.$inferSelect;

// Challenge Schema Types
export const insertChallengeSchema = createInsertSchema(challenges).pick({
  title: true,
  description: true,
  instructions: true,
  difficulty: true,
  disasterType: true,
  imageUrl: true,
  pointsValue: true,
  learningOutcomes: true,
  status: true,
  startDate: true,
  endDate: true,
  createdBy: true
});

export const insertChallengeResourceSchema = createInsertSchema(challengeResources).pick({
  challengeId: true,
  title: true,
  resourceType: true,
  url: true,
  description: true
});

export const insertUserChallengeProgressSchema = createInsertSchema(userChallengeProgress).pick({
  userId: true,
  challengeId: true,
  status: true,
  startedAt: true,
  completedAt: true,
  verifiedAt: true,
  pointsEarned: true,
  submissionNotes: true,
  submissionUrl: true
});

export const insertUserPointsSchema = createInsertSchema(userPoints).pick({
  userId: true,
  totalPoints: true,
  weeklyPoints: true,
  monthlyPoints: true,
  badgesEarned: true,
  challengesCompleted: true
});

export const insertChallengeDiscussionSchema = createInsertSchema(challengeDiscussions).pick({
  challengeId: true,
  userId: true,
  content: true,
  parentId: true,
  pinned: true
});

export type Challenge = typeof challenges.$inferSelect;
export type InsertChallenge = z.infer<typeof insertChallengeSchema>;

export type ChallengeResource = typeof challengeResources.$inferSelect;
export type InsertChallengeResource = z.infer<typeof insertChallengeResourceSchema>;

export type UserChallengeProgress = typeof userChallengeProgress.$inferSelect;
export type InsertUserChallengeProgress = z.infer<typeof insertUserChallengeProgressSchema>;

export type UserPoints = typeof userPoints.$inferSelect;
export type InsertUserPoints = z.infer<typeof insertUserPointsSchema>;

export type ChallengeDiscussion = typeof challengeDiscussions.$inferSelect;
export type InsertChallengeDiscussion = z.infer<typeof insertChallengeDiscussionSchema>;