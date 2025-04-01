import { users, type User, type InsertUser, 
  sentimentPosts, type SentimentPost, type InsertSentimentPost,
  disasterEvents, type DisasterEvent, type InsertDisasterEvent,
  analyzedFiles, type AnalyzedFile, type InsertAnalyzedFile,
  sessions, type LoginUser,
  profileImages, type ProfileImage, type InsertProfileImage,
  sentimentFeedback, type SentimentFeedback, type InsertSentimentFeedback,
  trainingExamples, type TrainingExample, type InsertTrainingExample
} from "@shared/schema";
import { db } from "./db";
import { eq } from "drizzle-orm";
import bcrypt from "bcryptjs";
import crypto from "crypto";

export interface IStorage {
  // User Management
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  loginUser(credentials: LoginUser): Promise<User | null>;
  createSession(userId: number): Promise<string>;
  validateSession(token: string): Promise<User | null>;

  // Sentiment Analysis
  getSentimentPosts(): Promise<SentimentPost[]>;
  getSentimentPostsByFileId(fileId: number): Promise<SentimentPost[]>;
  createSentimentPost(post: InsertSentimentPost): Promise<SentimentPost>;
  createManySentimentPosts(posts: InsertSentimentPost[]): Promise<SentimentPost[]>;
  deleteSentimentPost(id: number): Promise<void>;
  deleteAllSentimentPosts(): Promise<void>;
  deleteSentimentPostsByFileId(fileId: number): Promise<void>;

  // Disaster Events
  getDisasterEvents(): Promise<DisasterEvent[]>;
  createDisasterEvent(event: InsertDisasterEvent): Promise<DisasterEvent>;
  deleteDisasterEvent(id: number): Promise<void>;
  deleteAllDisasterEvents(): Promise<void>;

  // File Analysis
  getAnalyzedFiles(): Promise<AnalyzedFile[]>;
  getAnalyzedFile(id: number): Promise<AnalyzedFile | undefined>;
  createAnalyzedFile(file: InsertAnalyzedFile): Promise<AnalyzedFile>;
  deleteAnalyzedFile(id: number): Promise<void>;
  deleteAllAnalyzedFiles(): Promise<void>;
  updateFileMetrics(fileId: number, metrics: any): Promise<void>;

  // Profile Images
  getProfileImages(): Promise<ProfileImage[]>;
  createProfileImage(profile: InsertProfileImage): Promise<ProfileImage>;
  
  // Sentiment Feedback Training
  getSentimentFeedback(): Promise<SentimentFeedback[]>;
  getUntrainedFeedback(): Promise<SentimentFeedback[]>;
  submitSentimentFeedback(feedback: InsertSentimentFeedback): Promise<SentimentFeedback>;
  markFeedbackAsTrained(id: number): Promise<void>;
  
  // Training Examples Management
  getTrainingExamples(): Promise<TrainingExample[]>;
  getTrainingExampleByText(text: string): Promise<TrainingExample | undefined>;
  createTrainingExample(example: InsertTrainingExample): Promise<TrainingExample>;
  updateTrainingExample(id: number, sentiment: string): Promise<TrainingExample>;
  deleteTrainingExample(id: number): Promise<void>;

  // Delete All Data
  deleteAllData(): Promise<void>;
}

export class DatabaseStorage implements IStorage {
  // User Management
  async getUser(id: number): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user;
  }

  async createUser(insertUser: Omit<InsertUser, "confirmPassword">): Promise<User> {
    const hashedPassword = await bcrypt.hash(insertUser.password, 10);
    const [user] = await db.insert(users).values({
      ...insertUser,
      password: hashedPassword
    }).returning();
    return user;
  }

  async loginUser(credentials: LoginUser): Promise<User | null> {
    const user = await this.getUserByUsername(credentials.username);
    if (!user) return null;

    const valid = await bcrypt.compare(credentials.password, user.password);
    if (!valid) return null;

    return user;
  }

  async createSession(userId: number): Promise<string> {
    const token = crypto.randomBytes(32).toString('hex');
    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + 7); // 7 days expiration

    await db.insert(sessions).values({
      userId,
      token,
      expiresAt
    });

    return token;
  }

  async validateSession(token: string): Promise<User | null> {
    const [session] = await db.select()
      .from(sessions)
      .where(eq(sessions.token, token));

    if (!session || new Date() > session.expiresAt) {
      return null;
    }

    const user = await this.getUser(session.userId);
    return user || null;
  }

  // Sentiment Analysis
  async getSentimentPosts(): Promise<SentimentPost[]> {
    return db.select().from(sentimentPosts);
  }

  async getSentimentPostsByFileId(fileId: number): Promise<SentimentPost[]> {
    return db.select()
      .from(sentimentPosts)
      .where(eq(sentimentPosts.fileId, fileId));
  }

  async createSentimentPost(post: InsertSentimentPost): Promise<SentimentPost> {
    const [result] = await db.insert(sentimentPosts)
      .values(post)
      .returning();
    return result;
  }

  async createManySentimentPosts(posts: InsertSentimentPost[]): Promise<SentimentPost[]> {
    return db.insert(sentimentPosts)
      .values(posts)
      .returning();
  }

  async deleteSentimentPost(id: number): Promise<void> {
    await db.delete(sentimentPosts)
      .where(eq(sentimentPosts.id, id));
  }

  async deleteSentimentPostsByFileId(fileId: number): Promise<void> {
    await db.delete(sentimentPosts)
      .where(eq(sentimentPosts.fileId, fileId));
  }

  // Disaster Events
  async getDisasterEvents(): Promise<DisasterEvent[]> {
    return db.select().from(disasterEvents);
  }

  async createDisasterEvent(event: InsertDisasterEvent): Promise<DisasterEvent> {
    const [result] = await db.insert(disasterEvents)
      .values(event)
      .returning();
    return result;
  }
  
  async deleteDisasterEvent(id: number): Promise<void> {
    await db.delete(disasterEvents)
      .where(eq(disasterEvents.id, id));
  }

  // File Analysis
  async getAnalyzedFiles(): Promise<AnalyzedFile[]> {
    return db.select().from(analyzedFiles);
  }

  async getAnalyzedFile(id: number): Promise<AnalyzedFile | undefined> {
    const [file] = await db.select()
      .from(analyzedFiles)
      .where(eq(analyzedFiles.id, id));
    return file;
  }

  async createAnalyzedFile(file: InsertAnalyzedFile): Promise<AnalyzedFile> {
    const [result] = await db.insert(analyzedFiles)
      .values({
        ...file,
        evaluationMetrics: JSON.stringify(file.evaluationMetrics || null)
      })
      .returning();
    return result;
  }

  async deleteAnalyzedFile(id: number): Promise<void> {
    await this.deleteSentimentPostsByFileId(id);
    await db.delete(analyzedFiles)
      .where(eq(analyzedFiles.id, id));
  }
  async updateFileMetrics(fileId: number, metrics: any): Promise<void> {
    await db.update(analyzedFiles)
      .set({
        evaluationMetrics: JSON.stringify(metrics)
      })
      .where(eq(analyzedFiles.id, fileId));
  }

  // Delete functions
  async deleteAllSentimentPosts(): Promise<void> {
    await db.delete(sentimentPosts);
  }

  async deleteAllDisasterEvents(): Promise<void> {
    await db.delete(disasterEvents);
  }

  async deleteAllAnalyzedFiles(): Promise<void> {
    await db.delete(analyzedFiles);
  }

  // Profile Images
  async getProfileImages(): Promise<ProfileImage[]> {
    return db.select().from(profileImages);
  }

  async createProfileImage(profile: InsertProfileImage): Promise<ProfileImage> {
    const [result] = await db.insert(profileImages)
      .values(profile)
      .returning();
    return result;
  }

  // Sentiment Feedback Training 
  async getSentimentFeedback(): Promise<SentimentFeedback[]> {
    return db.select().from(sentimentFeedback);
  }

  async getUntrainedFeedback(): Promise<SentimentFeedback[]> {
    return db.select()
      .from(sentimentFeedback)
      .where(eq(sentimentFeedback.trainedOn, false));
  }

  async submitSentimentFeedback(feedback: InsertSentimentFeedback): Promise<SentimentFeedback> {
    const [result] = await db.insert(sentimentFeedback)
      .values(feedback)
      .returning();
    return result;
  }

  async markFeedbackAsTrained(id: number): Promise<void> {
    await db.update(sentimentFeedback)
      .set({ trainedOn: true })
      .where(eq(sentimentFeedback.id, id));
  }

  // Training Examples
  async getTrainingExamples(): Promise<TrainingExample[]> {
    return db.select().from(trainingExamples);
  }

  async getTrainingExampleByText(text: string): Promise<TrainingExample | undefined> {
    // Generate text_key similar to how it's done in Python backend
    const textWords = text.toLowerCase().match(/\b\w+\b/g) || [];
    const textKey = textWords.join(' ');
    
    const [example] = await db.select()
      .from(trainingExamples)
      .where(eq(trainingExamples.textKey, textKey));
    
    return example;
  }

  async createTrainingExample(example: InsertTrainingExample): Promise<TrainingExample> {
    // Attempt to find an existing entry first
    const existingExample = await this.getTrainingExampleByText(example.text);
    
    if (existingExample) {
      // If example already exists, update it rather than creating a new one
      return this.updateTrainingExample(existingExample.id, example.sentiment);
    }
    
    // Create a new training example
    const [result] = await db.insert(trainingExamples)
      .values(example)
      .returning();
    
    return result;
  }

  async updateTrainingExample(id: number, sentiment: string): Promise<TrainingExample> {
    const [result] = await db.update(trainingExamples)
      .set({ 
        sentiment, 
        updatedAt: new Date() 
      })
      .where(eq(trainingExamples.id, id))
      .returning();
    
    return result;
  }

  async deleteTrainingExample(id: number): Promise<void> {
    await db.delete(trainingExamples)
      .where(eq(trainingExamples.id, id));
  }

  async deleteAllData(): Promise<void> {
    await db.delete(sentimentFeedback);
    await db.delete(trainingExamples);
    await this.deleteAllSentimentPosts();
    await this.deleteAllDisasterEvents();
    await this.deleteAllAnalyzedFiles();
  }
}

export const storage = new DatabaseStorage();