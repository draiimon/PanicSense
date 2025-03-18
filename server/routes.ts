import express, { type Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from 'ws';
import { storage } from "./storage";
import path from "path";
import multer from "multer";
import { pythonService } from "./python-service";
import { insertSentimentPostSchema, insertAnalyzedFileSchema } from "@shared/schema";
import { EventEmitter } from 'events';

// Track upload progress
const uploadProgressMap = new Map<string, {
  processed: number;
  total: number;
  stage: string;
  timestamp: number;
}>();

// Track connected WebSocket clients
const connectedClients = new Set<WebSocket>();

// Configure multer for file uploads
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024
  },
  fileFilter: (req, file, cb) => {
    if (file.originalname.toLowerCase().endsWith('.csv')) {
      cb(null, true);
    } else {
      cb(new Error('Only CSV files are allowed'));
    }
  }
});

// Function to broadcast progress to all connected clients
function broadcastProgress(data: any) {
  const message = JSON.stringify({
    type: 'progress',
    progress: data
  });

  connectedClients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
}

export async function registerRoutes(app: Express): Promise<Server> {
  // Serve static files from attached_assets
  app.use('/assets', express.static(path.join(process.cwd(), 'attached_assets')));

  // Create HTTP server
  const httpServer = createServer(app);

  // Create WebSocket server
  const wss = new WebSocketServer({ 
    server: httpServer,
    path: '/ws'  
  });

  // WebSocket connection handler
  wss.on('connection', (ws: WebSocket) => {
    console.log('New WebSocket client connected');
    connectedClients.add(ws);

    // Handle client disconnection
    ws.on('close', () => {
      console.log('WebSocket client disconnected');
      connectedClients.delete(ws);
    });
  });


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
  const generateDisasterEvents = async (posts: any[]): Promise<void> => {
    if (posts.length === 0) return;

    // Group posts by day and disaster type
    const postsByDayAndType: {[key: string]: {
      posts: any[],
      count: number,
      sentiments: {[key: string]: number},
      type: string,
      location: string | null
    }} = {};

    // Group posts by day and disaster type
    for (const post of posts) {
      if (!post.disasterType) continue;

      const day = new Date(post.timestamp).toISOString().split('T')[0];
      const key = `${day}-${post.disasterType}`;

      if (!postsByDayAndType[key]) {
        postsByDayAndType[key] = {
          posts: [],
          count: 0,
          sentiments: {},
          type: post.disasterType,
          location: null
        };
      }

      postsByDayAndType[key].posts.push(post);
      postsByDayAndType[key].count++;

      // Track location with most occurrences
      if (post.location) {
        postsByDayAndType[key].location = post.location;
      }

      // Count sentiment occurrences
      const sentiment = post.sentiment;
      postsByDayAndType[key].sentiments[sentiment] = (postsByDayAndType[key].sentiments[sentiment] || 0) + 1;
    }

    // Process each group with sufficient posts (at least 3)
    for (const [key, data] of Object.entries(postsByDayAndType)) {
      if (data.count < 3) continue;

      // Find dominant sentiment and its change description
      let maxCount = 0;
      let dominantSentiment: string | null = null;
      let sentimentDescription = '';

      for (const [sentiment, count] of Object.entries(data.sentiments)) {
        if (count > maxCount) {
          maxCount = count;
          dominantSentiment = sentiment;
        }
      }

      // Create meaningful sentiment change descriptions
      if (dominantSentiment) {
        const percentage = Math.round((maxCount / data.count) * 100);
        if (percentage > 60) {
          switch(dominantSentiment) {
            case 'Fear/Anxiety':
              sentimentDescription = 'Fear/Anxiety sentiment spike';
              break;
            case 'Panic':
              sentimentDescription = 'Panic sentiment spike';
              break;
            case 'Neutral':
              sentimentDescription = 'Neutral sentiment trend';
              break;
            case 'Relief':
              sentimentDescription = 'Relief sentiment increase';
              break;
            case 'Disbelief':
              sentimentDescription = 'Disbelief sentiment surge';
              break;
            default:
              sentimentDescription = `${dominantSentiment} sentiment trend`;
          }
        } else {
          sentimentDescription = 'Mixed sentiment patterns';
        }
      }

      // Find most relevant sample content matching the disaster type
      const relevantPost = data.posts.find(post => 
        post.text.toLowerCase().includes(data.type.toLowerCase()) ||
        post.sentiment === dominantSentiment
      ) || data.posts[0];

      // Create the disaster event with improved description
      await storage.createDisasterEvent({
        name: `${data.type} Incident on ${new Date(key.split('-')[0]).toLocaleDateString()}`,
        description: `Based on ${data.count} social media reports. Sample content: ${relevantPost.text}`,
        timestamp: new Date(key.split('-')[0]),
        location: data.location,
        type: data.type,
        sentimentImpact: sentimentDescription
      });
    }
  };

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

  // File upload endpoint
  app.post('/api/upload-csv', upload.single('file'), async (req: Request, res: Response) => {
    const sessionId = req.headers['x-session-id'] as string;
    if (!sessionId) {
      return res.status(400).json({ error: "Session ID is required" });
    }

    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      const fileBuffer = req.file.buffer;
      const originalFilename = req.file.originalname;
      const fileContent = fileBuffer.toString('utf-8');
      const totalRecords = fileContent.split('\n').length - 1;

      // Initialize progress tracking
      uploadProgressMap.set(sessionId, {
        processed: 0,
        total: totalRecords,
        stage: `Initializing analysis for ${totalRecords} records...`,
        timestamp: Date.now()
      });

      // Progress callback for Python service
      const updateProgress = (processed: number, stage: string) => {
        const progress = {
          processed,
          total: totalRecords,
          stage,
          timestamp: Date.now()
        };

        // Update progress map
        uploadProgressMap.set(sessionId, progress);

        // Broadcast progress
        broadcastProgress(progress);
      };

      // Process the file
      const { data, storedFilename } = await pythonService.processCSV(
        fileBuffer,
        originalFilename,
        updateProgress
      );

      // Filter results
      const filteredResults = data.results.filter(post => 
        post.text.length >= 9 && 
        post.explanation && 
        post.disasterType && 
        post.disasterType !== "Not Specified"
      );

      // Save analyzed file
      const analyzedFile = await storage.createAnalyzedFile(
        insertAnalyzedFileSchema.parse({
          originalName: originalFilename,
          storedName: storedFilename,
          recordCount: filteredResults.length,
          evaluationMetrics: data.metrics
        })
      );

      // Save sentiment posts
      const sentimentPosts = await Promise.all(
        filteredResults.map(post => 
          storage.createSentimentPost(
            insertSentimentPostSchema.parse({
              text: post.text,
              timestamp: new Date(post.timestamp),
              source: post.source,
              language: post.language,
              sentiment: post.sentiment,
              confidence: post.confidence,
              location: post.location || null,
              disasterType: post.disasterType || null,
              fileId: analyzedFile.id
            })
          )
        )
      );

      // Final progress update
      const finalProgress = {
        processed: totalRecords,
        total: totalRecords,
        stage: 'Analysis complete',
        timestamp: Date.now()
      };
      uploadProgressMap.set(sessionId, finalProgress);
      broadcastProgress(finalProgress);

      // Clean up progress tracking
      setTimeout(() => {
        uploadProgressMap.delete(sessionId);
      }, 5000);

      res.json({
        file: analyzedFile,
        posts: sentimentPosts,
        metrics: data.metrics
      });
    } catch (error) {
      console.error("Error processing CSV:", error);

      const errorProgress = {
        processed: 0,
        total: 0,
        stage: 'Error: ' + (error instanceof Error ? error.message : String(error)),
        timestamp: Date.now()
      };
      uploadProgressMap.set(sessionId, errorProgress);
      broadcastProgress(errorProgress);

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

        // Check if this is a disaster-related post before saving
        // We consider text disaster-related if:
        // 1. It has a specific disaster type that's not "NONE"
        // 2. OR it has a specific location AND a sentiment that's not Neutral
        // 3. OR it has Fear/Anxiety or Panic sentiment which strongly suggests disaster context
        const isDisasterRelated = (
          (result.disasterType && result.disasterType !== "NONE" && result.disasterType !== "Not Specified") ||
          (result.location && result.sentiment !== "Neutral") ||
          ["Panic", "Fear/Anxiety"].includes(result.sentiment)
        );

        let sentimentPost;

        // Only save to database if it's disaster-related
        if (isDisasterRelated) {
          sentimentPost = await storage.createSentimentPost(
            insertSentimentPostSchema.parse({
              text,
              timestamp: new Date(),
              source,
              language: result.language,
              sentiment: result.sentiment,
              confidence: result.confidence,
              explanation: result.explanation,
              location: result.location || null,
              disasterType: result.disasterType || null,
              fileId: null
            })
          );

          return res.json({ 
            post: sentimentPost, 
            saved: true,
            message: "Disaster-related content detected and saved to database."
          });
        } else {
          // For non-disaster content, return the analysis but don't save it
          sentimentPost = {
            id: -1, 
            text,
            timestamp: new Date().toISOString(),
            source: 'Manual Input (Not Saved - Non-Disaster)',
            language: result.language,
            sentiment: result.sentiment,
            confidence: result.confidence,
            location: result.location,
            disasterType: result.disasterType,
            explanation: result.explanation,
            fileId: null
          };

          return res.json({ 
            post: sentimentPost, 
            saved: false,
            message: "Non-disaster content detected. Analysis shown but not saved to database."
          });
        }
      }

      // Process multiple texts
      const processResults = await Promise.all(texts.map(async (textItem: string) => {
        const result = await pythonService.analyzeSentiment(textItem);

        // Check if this is a disaster-related post
        const isDisasterRelated = (
          (result.disasterType && result.disasterType !== "NONE" && result.disasterType !== "Not Specified") ||
          (result.location && result.sentiment !== "Neutral") ||
          ["Panic", "Fear/Anxiety"].includes(result.sentiment)
        );

        if (isDisasterRelated) {
          // Only save disaster-related content
          const post = await storage.createSentimentPost(
            insertSentimentPostSchema.parse({
              text: textItem,
              timestamp: new Date(),
              source,
              language: result.language,
              sentiment: result.sentiment,
              confidence: result.confidence,
              explanation: result.explanation,
              location: result.location || null,
              disasterType: result.disasterType || null,
              fileId: null
            })
          );
          return { post, saved: true };
        } else {
          // Return analysis but don't save
          return { 
            post: {
              id: -1,
              text: textItem,
              timestamp: new Date().toISOString(),
              source: 'Manual Input (Not Saved - Non-Disaster)',
              language: result.language,
              sentiment: result.sentiment,
              confidence: result.confidence,
              location: result.location,
              disasterType: result.disasterType,
              explanation: result.explanation,
              fileId: null
            }, 
            saved: false 
          };
        }
      }));

      // Extract just the saved posts for disaster event generation
      const savedPosts = processResults
        .filter(item => item.saved)
        .map(item => item.post);

      // Generate disaster events from the saved posts if we have at least 3
      if (savedPosts.length >= 3) {
        await generateDisasterEvents(savedPosts);
      }

      res.json({
        results: processResults,
        savedCount: savedPosts.length,
        skippedCount: processResults.length - savedPosts.length,
        message: `Processed ${processResults.length} texts. Saved ${savedPosts.length} disaster-related posts. Skipped ${processResults.length - savedPosts.length} non-disaster posts.`
      });
    } catch (error) {
      res.status(500).json({ 
        error: "Failed to analyze text",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  // Delete all data endpoint
  app.delete('/api/delete-all-data', async (req: Request, res: Response) => {
    try {
      // Delete all data
      await storage.deleteAllData();

      res.json({ 
        success: true, 
        message: "All data has been deleted successfully"
      });
    } catch (error) {
      res.status(500).json({ 
        error: "Failed to delete all data",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  // Delete specific sentiment post endpoint
  app.delete('/api/sentiment-posts/:id', async (req: Request, res: Response) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({ error: "Invalid post ID" });
      }

      await storage.deleteSentimentPost(id);

      res.json({ 
        success: true, 
        message: `Sentiment post with ID ${id} has been deleted successfully`
      });
    } catch (error) {
      res.status(500).json({ 
        error: "Failed to delete sentiment post",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  // Delete analyzed file endpoint (deletes the file and all associated sentiment posts)
  app.delete('/api/analyzed-files/:id', async (req: Request, res: Response) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({ error: "Invalid file ID" });
      }

      // Check if file exists
      const file = await storage.getAnalyzedFile(id);
      if (!file) {
        return res.status(404).json({ error: "File not found" });
      }

      // Delete the file and all associated sentiment posts
      await storage.deleteAnalyzedFile(id);

      res.json({ 
        success: true, 
        message: `Deleted file "${file.originalName}" and all its associated sentiment posts`
      });
    } catch (error) {
      res.status(500).json({ 
        error: "Failed to delete analyzed file",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  // Add CSV export endpoint with formatted columns
  app.get('/api/export-csv', async (req: Request, res: Response) => {
    try {
      const posts = await storage.getSentimentPosts();

      // Create CSV header
      const csvHeader = 'Text,Timestamp,Source,Location,Disaster,Sentiment,Confidence,Language\n';

      // Format each post as CSV row
      const csvRows = posts.map(post => {
        const row = [
          `"${post.text.replace(/"/g, '""')}"`,
          post.timestamp,
          post.source || '',
          post.location || '',
          post.disasterType || '',
          post.sentiment,
          post.confidence,
          post.language
        ];
        return row.join(',');
      }).join('\n');

      const csv = csvHeader + csvRows;

      // Set headers for CSV download
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', 'attachment; filename=disaster-sentiments.csv');

      res.send(csv);
    } catch (error) {
      res.status(500).json({ error: "Failed to export CSV" });
    }
  });

  return httpServer;
}