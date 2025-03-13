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

    // Initialize with sample data
    this.initializeSampleData();
  }

  private async initializeSampleData() {
    // Add a sample analyzed file
    const sampleFile = await this.createAnalyzedFile({
      originalName: "sample-data.csv",
      storedName: "sample-data.csv",
      recordCount: 100,
      evaluationMetrics: {
        accuracy: 0.85,
        precision: 0.82,
        recall: 0.88,
        f1Score: 0.85
      },
      timestamp: new Date()
    });

    // Add some sample sentiment posts
    const samplePosts = [
      {
        text: "Earthquake hit our area, but community is helping each other.",
        timestamp: new Date(),
        source: "Twitter",
        language: "en",
        sentiment: "Resilience",
        confidence: 0.89,
        location: "Manila",
        disasterType: "Earthquake",
        fileId: sampleFile.id
      },
      {
        text: "Flooding getting worse in downtown area.",
        timestamp: new Date(),
        source: "Facebook",
        language: "en",
        sentiment: "Fear/Anxiety",
        confidence: 0.92,
        location: "Cebu",
        disasterType: "Flood",
        fileId: sampleFile.id
      }
    ];

    await this.createManySentimentPosts(samplePosts);

    // Add a sample disaster event
    await this.createDisasterEvent({
      name: "Manila Earthquake 2025",
      description: "6.2 magnitude earthquake in Metro Manila area",
      timestamp: new Date(),
      location: "Manila",
      type: "Earthquake",
      sentimentImpact: "Mixed"
    });
  }

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
      timestamp: insertFile.timestamp || new Date()
    };
    this.analyzedFiles.set(id, file);
    return file;
  }
}

export const storage = new MemStorage();