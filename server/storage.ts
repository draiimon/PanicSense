import { users, type User, type InsertUser, 
  sentimentPosts, type SentimentPost, type InsertSentimentPost,
  disasterEvents, type DisasterEvent, type InsertDisasterEvent,
  analyzedFiles, type AnalyzedFile, type InsertAnalyzedFile,
  sessions, type LoginUser,
  profileImages, type ProfileImage, type InsertProfileImage,
  sentimentFeedback, type SentimentFeedback, type InsertSentimentFeedback,
  trainingExamples, type TrainingExample, type InsertTrainingExample,
  uploadSessions, type UploadSession, type InsertUploadSession
} from "@shared/schema";
import { db, pool } from "./db";
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
  
  // Upload Sessions Management
  getUploadSession(sessionId: string): Promise<UploadSession | undefined>;
  createUploadSession(session: InsertUploadSession): Promise<UploadSession>;
  updateUploadSession(sessionId: string, status: string, progress: any): Promise<UploadSession | undefined>;
  deleteUploadSession(sessionId: string): Promise<void>;

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
    try {
      // Try the regular way first
      return db.select().from(sentimentPosts) as Promise<SentimentPost[]>;
    } catch (error) {
      console.error("Error in getSentimentPosts:", error);
      
      // Fallback to a more explicit select of only known-safe columns
      const results = await db.select({
        id: sentimentPosts.id,
        text: sentimentPosts.text,
        timestamp: sentimentPosts.timestamp,
        source: sentimentPosts.source,
        language: sentimentPosts.language,
        sentiment: sentimentPosts.sentiment,
        confidence: sentimentPosts.confidence,
        location: sentimentPosts.location,
        disasterType: sentimentPosts.disasterType,
        fileId: sentimentPosts.fileId,
        explanation: sentimentPosts.explanation,
        processedBy: sentimentPosts.processedBy
        // Deliberately omitting aiTrustMessage
      }).from(sentimentPosts);
      
      // Add missing aiTrustMessage field to each result (with null value)
      return results.map(post => ({
        ...post,
        aiTrustMessage: null
      })) as SentimentPost[];
    }
  }

  async getSentimentPostsByFileId(fileId: number): Promise<SentimentPost[]> {
    try {
      // Try the regular way first
      return db.select()
        .from(sentimentPosts)
        .where(eq(sentimentPosts.fileId, fileId)) as Promise<SentimentPost[]>;
    } catch (error) {
      console.error("Error in getSentimentPostsByFileId:", error);
      
      // Fallback to a more explicit select of only known-safe columns
      const results = await db.select({
        id: sentimentPosts.id,
        text: sentimentPosts.text,
        timestamp: sentimentPosts.timestamp,
        source: sentimentPosts.source,
        language: sentimentPosts.language,
        sentiment: sentimentPosts.sentiment,
        confidence: sentimentPosts.confidence,
        location: sentimentPosts.location,
        disasterType: sentimentPosts.disasterType,
        fileId: sentimentPosts.fileId,
        explanation: sentimentPosts.explanation,
        processedBy: sentimentPosts.processedBy
        // Deliberately omitting aiTrustMessage
      })
      .from(sentimentPosts)
      .where(eq(sentimentPosts.fileId, fileId));
      
      // Add missing aiTrustMessage field to each result (with null value)
      return results.map(post => ({
        ...post,
        aiTrustMessage: null
      })) as SentimentPost[];
    }
  }

  async createSentimentPost(post: InsertSentimentPost): Promise<SentimentPost> {
    try {
      // Create a safe copy without potentially problematic fields
      const safeSentimentPost: any = { 
        text: post.text,
        timestamp: post.timestamp,
        source: post.source,
        language: post.language,
        sentiment: post.sentiment,
        confidence: post.confidence,
        location: post.location,
        disasterType: post.disasterType,
        fileId: post.fileId,
        explanation: post.explanation,
        processedBy: post.processedBy
        // Deliberately omitting aiTrustMessage to avoid error if column doesn't exist
      };
      
      // Only include aiTrustMessage if it's present in the post and not undefined/null
      if (post.aiTrustMessage) {
        try {
          // Try to insert with aiTrustMessage first
          const [result] = await db.insert(sentimentPosts)
            .values({...safeSentimentPost, aiTrustMessage: post.aiTrustMessage})
            .returning();
          return result;
        } catch (err) {
          // If it fails due to missing column, fall back to insert without it
          console.log("Warning: aiTrustMessage column missing. Using fallback method.");
          const [result] = await db.insert(sentimentPosts)
            .values(safeSentimentPost)
            .returning();
          return result;
        }
      } else {
        // Regular insert without aiTrustMessage
        const [result] = await db.insert(sentimentPosts)
          .values(safeSentimentPost)
          .returning();
        return result;
      }
    } catch (error) {
      console.error("Error in createSentimentPost:", error);
      // Last resort - force insert with basic fields only
      const basicPost = {
        text: post.text,
        timestamp: post.timestamp || new Date(),
        source: post.source || "Unknown",
        language: post.language || "Unknown",
        sentiment: post.sentiment,
        confidence: post.confidence
      };
      
      const [result] = await db.insert(sentimentPosts)
        .values(basicPost)
        .returning();
      return result;
    }
  }

  async createManySentimentPosts(posts: InsertSentimentPost[]): Promise<SentimentPost[]> {
    try {
      // Create safe copies without potentially problematic fields
      const safePosts = posts.map(post => ({
        text: post.text,
        timestamp: post.timestamp,
        source: post.source,
        language: post.language,
        sentiment: post.sentiment,
        confidence: post.confidence,
        location: post.location,
        disasterType: post.disasterType,
        fileId: post.fileId,
        explanation: post.explanation,
        processedBy: post.processedBy
        // Deliberately omitting aiTrustMessage to avoid error if column doesn't exist
      }));
      
      return db.insert(sentimentPosts)
        .values(safePosts)
        .returning();
    } catch (error) {
      console.error("Error in createManySentimentPosts:", error);
      
      // Last resort - force insert with basic fields only
      const basicPosts = posts.map(post => ({
        text: post.text,
        timestamp: post.timestamp || new Date(),
        source: post.source || "Unknown",
        language: post.language || "Unknown",
        sentiment: post.sentiment,
        confidence: post.confidence
      }));
      
      return db.insert(sentimentPosts)
        .values(basicPosts)
        .returning();
    }
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
    try {
      // First delete all the sentiment posts that belong to this file
      console.log(`Deleting all sentiment posts for file ID ${id}`);
      await this.deleteSentimentPostsByFileId(id);
      
      // Then delete any upload sessions that reference this file
      try {
        console.log(`Deleting upload sessions for file ID ${id}`);
        await db.delete(uploadSessions)
          .where(eq(uploadSessions.fileId, id));
      } catch (uploadSessionError) {
        console.error(`Error deleting upload sessions for file ID ${id}:`, uploadSessionError);
        // Continue with the file deletion anyway
      }
      
      // Then delete the analyzed file record
      console.log(`Deleting analyzed file with ID ${id}`);
      await db.delete(analyzedFiles)
        .where(eq(analyzedFiles.id, id));
        
      console.log(`Successfully deleted file ID ${id} and all related posts and sessions`);
    } catch (error) {
      console.error(`Error in cascading delete of file ID ${id}:`, error);
      throw error; // Re-throw the error to be handled by the caller
    }
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
    try {
      return db.select().from(trainingExamples);
    } catch (error) {
      console.error("Error in getTrainingExamples (table may not exist):", error);
      // Return empty array if table doesn't exist
      return [];
    }
  }

  async getTrainingExampleByText(text: string): Promise<TrainingExample | undefined> {
    try {
      // Generate text_key similar to how it's done in Python backend
      const textWords = text.toLowerCase().match(/\b\w+\b/g) || [];
      const textKey = textWords.join(' ');
      
      const [example] = await db.select()
        .from(trainingExamples)
        .where(eq(trainingExamples.textKey, textKey));
      
      return example;
    } catch (error) {
      console.error("Error in getTrainingExampleByText (table may not exist):", error);
      // Return undefined if table doesn't exist
      return undefined;
    }
  }

  async createTrainingExample(example: InsertTrainingExample): Promise<TrainingExample> {
    try {
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
    } catch (error) {
      console.error("Error in createTrainingExample (table may not exist):", error);
      // Return a mock example since the table doesn't exist
      return {
        id: -1,
        text: example.text,
        textKey: example.textKey,
        sentiment: example.sentiment,
        language: example.language,
        confidence: example.confidence || 0.95,
        createdAt: new Date(),
        updatedAt: new Date()
      };
    }
  }

  async updateTrainingExample(id: number, sentiment: string): Promise<TrainingExample> {
    try {
      const [result] = await db.update(trainingExamples)
        .set({ 
          sentiment, 
          updatedAt: new Date() 
        })
        .where(eq(trainingExamples.id, id))
        .returning();
      
      return result;
    } catch (error) {
      console.error("Error in updateTrainingExample (table may not exist):", error);
      // Return a mock example since the table doesn't exist
      return {
        id: id,
        text: "Example not saved - database error",
        textKey: "example not saved",
        sentiment: sentiment,
        language: "unknown",
        confidence: 0.95,
        createdAt: new Date(),
        updatedAt: new Date()
      };
    }
  }

  async deleteTrainingExample(id: number): Promise<void> {
    try {
      await db.delete(trainingExamples)
        .where(eq(trainingExamples.id, id));
    } catch (error) {
      console.error("Error in deleteTrainingExample (table may not exist):", error);
      // Just log the error, no need to throw
    }
  }

  // Upload Sessions Management
  async getUploadSession(sessionId: string): Promise<UploadSession | undefined> {
    try {
      const result = await pool.query(`
        SELECT * FROM upload_sessions WHERE session_id = $1
      `, [sessionId]);
      
      return result.rows[0];
    } catch (error) {
      console.error("Error in getUploadSession (table may not exist):", error);
      return undefined;
    }
  }

  async createUploadSession(session: InsertUploadSession): Promise<UploadSession> {
    try {
      // Import SERVER_START_TIMESTAMP or use fallback value
      let serverTimestamp = session.serverStartTimestamp;
      if (!serverTimestamp) {
        try {
          const { SERVER_START_TIMESTAMP } = await import('./index');
          serverTimestamp = SERVER_START_TIMESTAMP.toString();
        } catch (e) {
          // Fallback if import fails
          serverTimestamp = Date.now().toString();
        }
      }

      try {
        // Try using SQL instead of Drizzle to avoid column issues
        const result = await pool.query(`
          INSERT INTO upload_sessions 
            (session_id, status, progress, file_id, user_id, server_start_timestamp, created_at, updated_at)
          VALUES 
            ($1, $2, $3, $4, $5, $6, $7, $8)
          RETURNING *
        `, [
          session.sessionId,
          session.status || 'active',
          session.progress ? JSON.stringify(session.progress) : null,
          session.fileId || null,
          session.userId || null,
          serverTimestamp,
          new Date(),
          new Date()
        ]);

        return result.rows[0];
      } catch (innerError) {
        console.error("Error in direct SQL insert, trying without server_start_timestamp:", innerError);
        
        // Fallback: try inserting without server_start_timestamp if it doesn't exist
        const result = await pool.query(`
          INSERT INTO upload_sessions 
            (session_id, status, progress, file_id, user_id, created_at, updated_at)
          VALUES 
            ($1, $2, $3, $4, $5, $6, $7)
          RETURNING *
        `, [
          session.sessionId,
          session.status || 'active',
          session.progress ? JSON.stringify(session.progress) : null,
          session.fileId || null,
          session.userId || null,
          new Date(),
          new Date()
        ]);

        return result.rows[0];
      }
    } catch (error) {
      console.error("Error in createUploadSession (table may not exist):", error);
      // Return a mock session since the table might not exist
      return {
        id: -1,
        sessionId: session.sessionId,
        status: session.status || 'active',
        fileId: session.fileId || null,  // Ensure fileId is null if undefined
        progress: session.progress,
        createdAt: new Date(),
        updatedAt: new Date(),
        userId: session.userId || null,   // Ensure userId is null if undefined
        serverStartTimestamp: session.serverStartTimestamp || null  // Add this to match expected type
      };
    }
  }

  async updateUploadSession(sessionId: string, status: string, progress: any): Promise<UploadSession | undefined> {
    try {
      // Get the server timestamp 
      let serverTimestamp;
      try {
        const { SERVER_START_TIMESTAMP } = await import('./index');
        serverTimestamp = SERVER_START_TIMESTAMP.toString();
      } catch (e) {
        // Fallback if import fails
        serverTimestamp = Date.now().toString();
      }

      try {
        const result = await pool.query(`
          UPDATE upload_sessions
          SET status = $1, progress = $2, updated_at = $3, server_start_timestamp = $4
          WHERE session_id = $5
          RETURNING *
        `, [
          status,
          progress ? JSON.stringify(progress) : null,
          new Date(),
          serverTimestamp,
          sessionId
        ]);
        
        return result.rows[0];
      } catch (innerError) {
        console.error("Error in direct SQL update, trying without server_start_timestamp:", innerError);
        
        // Fallback: try updating without server_start_timestamp
        const result = await pool.query(`
          UPDATE upload_sessions
          SET status = $1, progress = $2, updated_at = $3
          WHERE session_id = $4
          RETURNING *
        `, [
          status,
          progress ? JSON.stringify(progress) : null,
          new Date(),
          sessionId
        ]);
        
        return result.rows[0];
      }
    } catch (error) {
      console.error("Error in updateUploadSession (table may not exist):", error);
      return undefined;
    }
  }

  async deleteUploadSession(sessionId: string): Promise<void> {
    try {
      await pool.query(`
        DELETE FROM upload_sessions WHERE session_id = $1
      `, [sessionId]);
    } catch (error) {
      console.error("Error in deleteUploadSession (table may not exist):", error);
      // Just log the error, no need to throw
    }
  }

  async deleteAllData(): Promise<void> {
    await db.delete(sentimentFeedback);
    await db.delete(trainingExamples);
    await db.delete(uploadSessions);
    await this.deleteAllSentimentPosts();
    await this.deleteAllDisasterEvents();
    await this.deleteAllAnalyzedFiles();
  }
}

export const storage = new DatabaseStorage();