import { users, type User, type InsertUser, 
  sentimentPosts, type SentimentPost, type InsertSentimentPost,
  disasterEvents, type DisasterEvent, type InsertDisasterEvent,
  analyzedFiles, type AnalyzedFile, type InsertAnalyzedFile 
} from "@shared/schema";

export interface IStorage {
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;

  getSentimentPosts(): Promise<SentimentPost[]>;
  getSentimentPostsByFileId(fileId: number): Promise<SentimentPost[]>;
  createSentimentPost(post: InsertSentimentPost): Promise<SentimentPost>;
  createManySentimentPosts(posts: InsertSentimentPost[]): Promise<SentimentPost[]>;

  getDisasterEvents(): Promise<DisasterEvent[]>;
  createDisasterEvent(event: InsertDisasterEvent): Promise<DisasterEvent>;

  getAnalyzedFiles(): Promise<AnalyzedFile[]>;
  getAnalyzedFile(id: number): Promise<AnalyzedFile | undefined>;
  createAnalyzedFile(file: InsertAnalyzedFile): Promise<AnalyzedFile>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private sentimentPosts: Map<number, SentimentPost>;
  private disasterEvents: Map<number, DisasterEvent>;
  private analyzedFiles: Map<number, AnalyzedFile>;

  private userCurrentId: number;
  private sentimentPostCurrentId: number;
  private disasterEventCurrentId: number;
  private analyzedFileCurrentId: number;

  constructor() {
    this.users = new Map();
    this.sentimentPosts = new Map();
    this.disasterEvents = new Map();
    this.analyzedFiles = new Map();

    this.userCurrentId = 1;
    this.sentimentPostCurrentId = 1;
    this.disasterEventCurrentId = 1;
    this.analyzedFileCurrentId = 1;
  }

  // User methods
  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.userCurrentId++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  // Sentiment Posts methods
  async getSentimentPosts(): Promise<SentimentPost[]> {
    return Array.from(this.sentimentPosts.values());
  }

  async getSentimentPostsByFileId(fileId: number): Promise<SentimentPost[]> {
    return Array.from(this.sentimentPosts.values()).filter(
      post => post.fileId === fileId
    );
  }

  async createSentimentPost(insertPost: InsertSentimentPost): Promise<SentimentPost> {
    const id = this.sentimentPostCurrentId++;
    const post: SentimentPost = { 
      ...insertPost, 
      id,
      timestamp: insertPost.timestamp || new Date()
    };
    this.sentimentPosts.set(id, post);
    return post;
  }

  async createManySentimentPosts(posts: InsertSentimentPost[]): Promise<SentimentPost[]> {
    return Promise.all(posts.map(post => this.createSentimentPost(post)));
  }

  // Disaster Events methods
  async getDisasterEvents(): Promise<DisasterEvent[]> {
    return Array.from(this.disasterEvents.values());
  }

  async createDisasterEvent(insertEvent: InsertDisasterEvent): Promise<DisasterEvent> {
    const id = this.disasterEventCurrentId++;
    const event: DisasterEvent = { 
      ...insertEvent, 
      id,
      timestamp: insertEvent.timestamp || new Date()
    };
    this.disasterEvents.set(id, event);
    return event;
  }

  // Analyzed Files methods
  async getAnalyzedFiles(): Promise<AnalyzedFile[]> {
    return Array.from(this.analyzedFiles.values());
  }

  async getAnalyzedFile(id: number): Promise<AnalyzedFile | undefined> {
    return this.analyzedFiles.get(id);
  }

  async createAnalyzedFile(insertFile: InsertAnalyzedFile): Promise<AnalyzedFile> {
    const id = this.analyzedFileCurrentId++;
    const file: AnalyzedFile = { 
      ...insertFile, 
      id,
      timestamp: new Date()
    };
    this.analyzedFiles.set(id, file);
    return file;
  }
}

export const storage = new MemStorage();
