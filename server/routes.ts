import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import multer from "multer";
import { pythonService } from "./python-service";
import { insertSentimentPostSchema, insertAnalyzedFileSchema } from "@shared/schema";
import { EventEmitter } from 'events';

// Configure multer for file uploads with improved performance
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024, // Increased to 50MB for faster batch processing
  },
  fileFilter: (req, file, cb) => {
    if (file.originalname.toLowerCase().endsWith('.csv')) {
      cb(null, true);
    } else {
      cb(new Error('Only CSV files are allowed'));
    }
  }
});

// Enhanced upload progress tracking with better performance
const uploadProgressMap = new Map<string, {
  processed: number;
  total: number;
  stage: string;
  timestamp: number;
  error?: string;
}>();

export async function registerRoutes(app: Express): Promise<Server> {
  // Add the SSE endpoint inside registerRoutes
  app.get('/api/upload-progress/:sessionId', (req: Request, res: Response) => {
    const sessionId = req.params.sessionId;

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    });

    const sendProgress = () => {
      const progress = uploadProgressMap.get(sessionId);
      if (progress) {
        const now = Date.now();
        if (now - progress.timestamp >= 50) { // Increased frequency for smoother updates
          res.write(`data: ${JSON.stringify({
            processed: progress.processed,
            total: progress.total,
            stage: progress.stage,
            percentage: Math.round((progress.processed / progress.total) * 100),
            error: progress.error
          })}\n\n`);
          progress.timestamp = now;
        }
      }
    };

    const progressInterval = setInterval(sendProgress, 50);

    req.on('close', () => {
      clearInterval(progressInterval);
      uploadProgressMap.delete(sessionId);
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

    // Group posts by day to identify patterns
    const postsByDay: {[key: string]: {
      posts: any[],
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

  // Update file upload endpoint with better progress tracking
  app.post('/api/upload-csv', upload.single('file'), async (req: Request, res: Response) => {
    let sessionId: string | undefined;
    let updateProgress: (processed: number, stage: string, error?: string) => void;

    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      sessionId = req.headers['x-session-id'] as string;
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }

      const fileBuffer = req.file.buffer;
      const originalFilename = req.file.originalname;
      const fileContent = fileBuffer.toString('utf-8');
      const totalRecords = fileContent.split('\n').length - 1;

      // Initialize progress tracking
      uploadProgressMap.set(sessionId, {
        processed: 0,
        total: totalRecords,
        stage: 'Starting analysis',
        timestamp: Date.now()
      });

      updateProgress = (processed: number, stage: string, error?: string) => {
        if (sessionId) {
          const progress = uploadProgressMap.get(sessionId);
          if (progress) {
            progress.processed = processed;
            progress.stage = stage;
            progress.timestamp = Date.now();
            progress.error = error;
          }
        }
      };

      const { data, storedFilename, recordCount } = await pythonService.processCSV(
        fileBuffer,
        originalFilename,
        updateProgress
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
              location: post.location || null,
              disasterType: post.disasterType || null,
              fileId: analyzedFile.id
            })
          )
        )
      );

      // Generate disaster events from the sentiment posts
      await generateDisasterEvents(sentimentPosts);

      // Final progress update
      if (sessionId && updateProgress) {
        updateProgress(totalRecords, 'Analysis complete');
      }

      res.json({
        file: analyzedFile,
        posts: sentimentPosts,
        metrics: data.metrics,
        sessionId
      });
    } catch (error) {
      console.error("Error processing CSV:", error);
      if (sessionId && updateProgress) {
        updateProgress(0, 'Error', error instanceof Error ? error.message : String(error));
      }
      res.status(500).json({ 
        error: "Failed to process CSV file",
        details: error instanceof Error ? error.message : String(error)
      });
    } finally {
      // Cleanup progress tracking after 5 seconds
      if (sessionId) {
        setTimeout(() => {
          uploadProgressMap.delete(sessionId);
        }, 5000);
      }
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
            id: -1, // Use negative ID to indicate this wasn't saved
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

  const httpServer = createServer(app);
  return httpServer;
}