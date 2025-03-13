import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import multer from "multer";
import { pythonService } from "./python-service";
import { insertSentimentPostSchema, insertAnalyzedFileSchema, insertDisasterEventSchema, type SentimentPost, type DisasterEvent } from "@shared/schema";
import fs from "fs";
import path from "path";
import os from "os";

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

export async function registerRoutes(app: Express): Promise<Server> {
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
      
      // Analyze text to determine disaster type
      const combinedText = data.posts.map(p => p.text.toLowerCase()).join(' ');
      let disasterType = "Undetermined";
      
      if (combinedText.includes('earthquake') || combinedText.includes('quake') || 
          combinedText.includes('lindol') || combinedText.includes('tremor')) {
        disasterType = "Earthquake";
      } else if (combinedText.includes('flood') || combinedText.includes('baha') || 
                combinedText.includes('water level')) {
        disasterType = "Flood";
      } else if (combinedText.includes('typhoon') || combinedText.includes('bagyo') || 
                combinedText.includes('storm')) {
        disasterType = "Typhoon";
      } else if (combinedText.includes('fire') || combinedText.includes('sunog')) {
        disasterType = "Fire";
      } else if (combinedText.includes('landslide') || combinedText.includes('guho')) {
        disasterType = "Landslide";
      } else if (combinedText.includes('volcanic') || combinedText.includes('volcano') || 
                combinedText.includes('bulkan') || combinedText.includes('ash')) {
        disasterType = "Volcanic Activity";
      }
      
      // Find potential location
      let location: string | null = null;
      
      // Common Philippines locations to check for in the text
      const locations = [
        'Manila', 'Quezon City', 'Cebu', 'Davao', 'Mindanao', 'Luzon',
        'Visayas', 'Palawan', 'Boracay', 'Baguio', 'Bohol', 'Iloilo',
        'Batangas', 'Zambales', 'Pampanga', 'Bicol', 'Leyte', 'Samar',
        'Pangasinan', 'Tarlac', 'Cagayan', 'Bulacan', 'Cavite', 'Laguna'
      ];
      
      for (const loc of locations) {
        if (combinedText.includes(loc.toLowerCase())) {
          location = loc;
          break;
        }
      }
      
      // Generate a summary from the posts
      const sampleTexts = data.posts.slice(0, 3).map(p => 
        p.text.length > 50 ? p.text.substring(0, 50) + '...' : p.text
      ).join(' | ');
      
      const description = `Based on ${data.count} social media reports. Sample content: ${sampleTexts}`;
      
      // Create the disaster event
      await storage.createDisasterEvent(
        insertDisasterEventSchema.parse({
          name: `${disasterType} Incident on ${new Date(day).toLocaleDateString()}`,
          description,
          timestamp: new Date(day),
          location,
          type: disasterType,
          sentimentImpact: dominantSentiment
        })
      );
    }
  };

  // API Routes
  
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

  // Upload and analyze CSV file
  app.post('/api/upload-csv', upload.single('file'), async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      const fileBuffer = req.file.buffer;
      const originalFilename = req.file.originalname;

      const { data, storedFilename, recordCount } = await pythonService.processCSV(fileBuffer, originalFilename);
      
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
            language: 'en', // Could be improved to detect language
            sentiment: result.sentiment,
            confidence: result.confidence,
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
            language: 'en', // Could be improved to detect language
            sentiment: result.sentiment,
            confidence: result.confidence,
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
