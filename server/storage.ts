import { users, type User, type InsertUser, 
  sentimentPosts, type SentimentPost, type InsertSentimentPost,
  disasterEvents, type DisasterEvent, type InsertDisasterEvent,
  analyzedFiles, type AnalyzedFile, type InsertAnalyzedFile,
  sessions, type LoginUser
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

  // Disaster Events
  getDisasterEvents(): Promise<DisasterEvent[]>;
  createDisasterEvent(event: InsertDisasterEvent): Promise<DisasterEvent>;

  // File Analysis
  getAnalyzedFiles(): Promise<AnalyzedFile[]>;
  getAnalyzedFile(id: number): Promise<AnalyzedFile | undefined>;
  createAnalyzedFile(file: InsertAnalyzedFile): Promise<AnalyzedFile>;
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
      .values(file)
      .returning();
    return result;
  }
}

export const storage = new DatabaseStorage();