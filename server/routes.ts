import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import multer from "multer";
import { pythonService } from "./python-service";
import { insertSentimentPostSchema, insertAnalyzedFileSchema, insertDisasterEventSchema, type SentimentPost, type DisasterEvent } from "@shared/schema";
import fs from "fs";
import path from "path";
import os from "os";
import { EventEmitter } from 'events';

// Configure multer for file uploads
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    // Accept any file with .csv extension regardless of mimetype
    if (file.originalname.toLowerCase().endsWith('.csv')) {
      cb(null, true);
    } else {
      cb(new Error('Only CSV files are allowed'));
    }
  }
});

// Create a global event emitter for upload progress
const uploadProgressEmitter = new EventEmitter();

export async function registerRoutes(app: Express): Promise<Server> {
  // Authentication Routes
  app.post('/api/auth/signup', async (req: Request, res: Response) => {
    try {
      const { username, password, email, fullName } = req.body;

      // Check if user already exists
      const existingUser = await storage.getUserByUsername(username);
      if (existingUser) {
        return res.status(400).json({ error: "Username already taken" });
      }

      // Create new user
      const user = await storage.createUser({
        username,
        password,
        email,
        fullName,
        role: 'user'
      });

      // Create session
      const token = await storage.createSession(user.id);

      res.json({ token });
    } catch (error) {
      res.status(500).json({ 
        error: "Failed to create user",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  app.post('/api/auth/login', async (req: Request, res: Response) => {
    try {
      const { username, password } = req.body;
      const user = await storage.loginUser({ username, password });

      if (!user) {
        return res.status(401).json({ error: "Invalid credentials" });
      }

      const token = await storage.createSession(user.id);
      res.json({ token });
    } catch (error) {
      res.status(500).json({ 
        error: "Login failed",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  app.get('/api/auth/me', async (req: Request, res: Response) => {
    try {
      const token = req.headers.authorization?.split(' ')[1];
      if (!token) {
        return res.status(401).json({ error: "No token provided" });
      }

      const user = await storage.validateSession(token);
      if (!user) {
        return res.status(401).json({ error: "Invalid or expired token" });
      }

      // Don't send password in response
      const { password, ...userWithoutPassword } = user;
      res.json(userWithoutPassword);
    } catch (error) {
      res.status(500).json({ 
        error: "Failed to get user info",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  // Helper function to generate disaster events from sentiment posts
  const generateDisasterEvents = async (posts: SentimentPost[]): Promise<void> => {
    if (posts.length === 0) return;

    // Group posts by day to identify patterns
    const postsByDay: {[key: string]: {
      posts: SentimentPost[],
      count: number,
      sentiments: {[key: string]: number}
    }} = {};

    // Group posts by day (YYYY-MM-DD)
    for (const post of posts) {
      const day = new Date(post.timestamp).toISOString().split('T')[0];

      if (!postsByDay[day]) {
        postsByDay[day] = {
          posts: [],
          count: 0,
          sentiments: {}
        };
      }

      postsByDay[day].posts.push(post);
      postsByDay[day].count++;

      // Count sentiment occurrences
      const sentiment = post.sentiment;
      postsByDay[day].sentiments[sentiment] = (postsByDay[day].sentiments[sentiment] || 0) + 1;
    }

    // Process each day with sufficient posts (at least 3)
    for (const [day, data] of Object.entries(postsByDay)) {
      if (data.count < 3) continue;

      // Find dominant sentiment
      let maxCount = 0;
      let dominantSentiment: string | null = null;

      for (const [sentiment, count] of Object.entries(data.sentiments)) {
        if (count > maxCount) {
          maxCount = count;
          dominantSentiment = sentiment;
        }
      }

      // Extract disaster type and location from text content
      const texts = data.posts.map(p => p.text.toLowerCase());
      let disasterType = null;
      let location = null;

      // Enhanced disaster type detection with more variations
      const disasterKeywords = {
        "Earthquake": ['lindol', 'earthquake', 'quake', 'tremor', 'lumindol', 'yugto', 'lindol na malakas', 'paglindol'],
        "Flood": ['baha', 'flood', 'pagbaha', 'pagbabaha', 'bumaha', 'tubig', 'binaha', 'napabaha', 'flash flood'],
        "Typhoon": ['bagyo', 'typhoon', 'storm', 'cyclone', 'hurricane', 'bagyong', 'unos', 'habagat', 'super typhoon'],
        "Fire": ['sunog', 'fire', 'nasunog', 'burning', 'apoy', 'silab', 'nagkasunog', 'wildfire', 'forest fire'],
        "Volcanic Eruption": ['bulkan', 'volcano', 'eruption', 'ash fall', 'lava', 'ashfall', 'bulkang', 'pumutok', 'sumabog'],
        "Landslide": ['landslide', 'pagguho', 'guho', 'mudslide', 'rockslide', 'avalanche', 'pagguho ng lupa', 'collapsed'],
        "Tsunami": ['tsunami', 'tidal wave', 'daluyong', 'alon', 'malalaking alon']
      };

      // Check each disaster type in texts
      for (const [type, keywords] of Object.entries(disasterKeywords)) {
        if (texts.some(text => keywords.some(keyword => text.includes(keyword)))) {
          disasterType = type;
          break;
        }
      }

      // Enhanced location detection with more Philippine locations
      const locations = [
        'Manila', 'Quezon City', 'Cebu', 'Davao', 'Mindanao', 'Luzon',
        'Visayas', 'Palawan', 'Boracay', 'Baguio', 'Bohol', 'Iloilo',
        'Batangas', 'Zambales', 'Pampanga', 'Bicol', 'Leyte', 'Samar',
        'Pangasinan', 'Tarlac', 'Cagayan', 'Bulacan', 'Cavite', 'Laguna',
        'Rizal', 'Marikina', 'Makati', 'Pasig', 'Taguig', 'Pasay', 'Mandaluyong',
        'Parañaque', 'Caloocan', 'Valenzuela', 'Muntinlupa', 'Malabon', 'Navotas',
        'San Juan', 'Las Piñas', 'Pateros', 'Nueva Ecija', 'Benguet', 'Albay',
        'Catanduanes', 'Sorsogon', 'Camarines Sur', 'Camarines Norte', 'Marinduque'
      ];

      // Try to find locations in text more aggressively
      for (const text of texts) {
        const textLower = text.toLowerCase();
        for (const loc of locations) {
          if (textLower.includes(loc.toLowerCase())) {
            location = loc;
            break;
          }
        }
        if (location) break;
      }

      if (disasterType && (location || dominantSentiment)) {
        // Create the disaster event
        await storage.createDisasterEvent({
          name: `${disasterType} Incident on ${new Date(day).toLocaleDateString()}`,
          description: `Based on ${data.count} social media reports. Sample content: ${data.posts[0].text}`,
          timestamp: new Date(day),
          location,
          type: disasterType,
          sentimentImpact: dominantSentiment || undefined
        });
      }
    }
  };

  // Add SSE endpoint for upload progress
  app.get('/api/upload-progress', (req: Request, res: Response) => {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    });

    const sendProgress = (progress: { processed: number; stage: string }) => {
      res.write(`data: ${JSON.stringify(progress)}\n\n`);
    };

    uploadProgressEmitter.on('progress', sendProgress);

    req.on('close', () => {
      uploadProgressEmitter.off('progress', sendProgress);
    });
  });

  // Get all sentiment posts
  app.get('/api/sentiment-posts', async (req: Request, res: Response) => {
    try {
      const posts = await storage.getSentimentPosts();
      res.json(posts);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch sentiment posts" });
    }
  });

  // Get sentiment posts by file id
  app.get('/api/sentiment-posts/file/:fileId', async (req: Request, res: Response) => {
    try {
      const fileId = parseInt(req.params.fileId);
      const posts = await storage.getSentimentPostsByFileId(fileId);
      res.json(posts);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch sentiment posts" });
    }
  });

  // Get all disaster events
  app.get('/api/disaster-events', async (req: Request, res: Response) => {
    try {
      const events = await storage.getDisasterEvents();
      res.json(events);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch disaster events" });
    }
  });

  // Get all analyzed files
  app.get('/api/analyzed-files', async (req: Request, res: Response) => {
    try {
      const files = await storage.getAnalyzedFiles();
      res.json(files);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch analyzed files" });
    }
  });

  // Get specific analyzed file
  app.get('/api/analyzed-files/:id', async (req: Request, res: Response) => {
    try {
      const id = parseInt(req.params.id);
      const file = await storage.getAnalyzedFile(id);
      
      if (!file) {
        return res.status(404).json({ error: "Analyzed file not found" });
      }
      
      res.json(file);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch analyzed file" });
    }
  });

  // Modify upload-csv endpoint to emit progress
  app.post('/api/upload-csv', upload.single('file'), async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      const fileBuffer = req.file.buffer;
      const originalFilename = req.file.originalname;

      // Read the file content to count total records
      const fileContent = fileBuffer.toString('utf-8');
      const totalRecords = fileContent.split('\n').length - 1; // -1 for header

      // Emit initial progress
      uploadProgressEmitter.emit('progress', {
        processed: 0,
        stage: 'Starting analysis'
      });

      const { data, storedFilename, recordCount } = await pythonService.processCSV(
        fileBuffer, 
        originalFilename,
        (processed: number, stage: string) => {
          uploadProgressEmitter.emit('progress', { processed, stage });
        }
      );

      // Save the analyzed file record
      const analyzedFile = await storage.createAnalyzedFile(
        insertAnalyzedFileSchema.parse({
          originalName: originalFilename,
          storedName: storedFilename,
          recordCount: recordCount,
          evaluationMetrics: data.metrics
        })
      );

      // Save all sentiment posts
      const sentimentPosts = await Promise.all(
        data.results.map(post => 
          storage.createSentimentPost(
            insertSentimentPostSchema.parse({
              text: post.text,
              timestamp: new Date(post.timestamp),
              source: post.source,
              language: post.language,
              sentiment: post.sentiment,
              confidence: post.confidence,
              location: null,
              disasterType: null,
              fileId: analyzedFile.id
            })
          )
        )
      );

      // Generate disaster events from the sentiment posts
      await generateDisasterEvents(sentimentPosts);

      // Emit completion progress
      uploadProgressEmitter.emit('progress', {
        processed: totalRecords,
        stage: 'Analysis complete'
      });

      res.json({
        file: analyzedFile,
        posts: sentimentPosts,
        metrics: data.metrics
      });
    } catch (error) {
      console.error("Error processing CSV:", error);
      res.status(500).json({ 
        error: "Failed to process CSV file",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  // Analyze text (single or batch)
  app.post('/api/analyze-text', async (req: Request, res: Response) => {
    try {
      const { text, texts, source = 'Manual Input' } = req.body;
      
      // Check if we have either a single text or an array of texts
      if (!text && (!texts || !Array.isArray(texts) || texts.length === 0)) {
        return res.status(400).json({ error: "No text provided. Send either 'text' or 'texts' array in the request body" });
      }
      
      // Process single text
      if (text) {
        const result = await pythonService.analyzeSentiment(text);
        
        // Save the sentiment post
        const sentimentPost = await storage.createSentimentPost(
          insertSentimentPostSchema.parse({
            text,
            timestamp: new Date(),
            source,
            language: result.language,
            sentiment: result.sentiment,
            confidence: result.confidence,
            explanation: result.explanation,
            location: null,
            disasterType: null,
            fileId: null
          })
        );
        
        return res.json({ post: sentimentPost });
      }
      
      // Process multiple texts
      const sentimentPromises = texts.map(async (textItem: string) => {
        const result = await pythonService.analyzeSentiment(textItem);
        
        return storage.createSentimentPost(
          insertSentimentPostSchema.parse({
            text: textItem,
            timestamp: new Date(),
            source,
            language: result.language,
            sentiment: result.sentiment,
            confidence: result.confidence,
            explanation: result.explanation,
            location: null,
            disasterType: null,
            fileId: null
          })
        );
      });
      
      const sentimentPosts = await Promise.all(sentimentPromises);
      
      // Generate disaster events from the new posts if we have at least 3
      if (sentimentPosts.length >= 3) {
        await generateDisasterEvents(sentimentPosts);
      }
      
      res.json({
        posts: sentimentPosts
      });
    } catch (error) {
      res.status(500).json({ 
        error: "Failed to analyze text",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}