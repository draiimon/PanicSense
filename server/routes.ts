import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import multer from "multer";
import { pythonService } from "./python-service";
import { insertSentimentPostSchema, insertAnalyzedFileSchema } from "@shared/schema";
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

  // Analyze single text
  app.post('/api/analyze-text', async (req: Request, res: Response) => {
    try {
      const { text } = req.body;
      
      if (!text) {
        return res.status(400).json({ error: "No text provided" });
      }

      const result = await pythonService.analyzeSentiment(text);
      
      // Save the sentiment post
      const sentimentPost = await storage.createSentimentPost(
        insertSentimentPostSchema.parse({
          text,
          timestamp: new Date(),
          source: 'Manual Input',
          language: 'en',
          sentiment: result.sentiment,
          confidence: result.confidence,
          location: null,
          disasterType: null,
          fileId: null
        })
      );

      res.json({
        post: sentimentPost
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
