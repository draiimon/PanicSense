import express, { type Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from 'ws';
import { storage } from "./storage";
import path from "path";
import multer from "multer";
import { pythonService, pythonConsoleMessages } from "./python-service";
import { insertSentimentPostSchema, insertAnalyzedFileSchema } from "@shared/schema";
import { usageTracker } from "./utils/usage-tracker";
import { EventEmitter } from 'events';

// Configure multer for file uploads with improved performance
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024, 
  },
  fileFilter: (req, file, cb) => {
    if (file.originalname.toLowerCase().endsWith('.csv')) {
      cb(null, true);
    } else {
      cb(new Error('Only CSV files are allowed'));
    }
  }
});

// Enhanced progress tracking with more details
const uploadProgressMap = new Map<string, {
  processed: number;
  total: number;
  stage: string;
  timestamp: number;
  batchNumber: number;
  totalBatches: number;
  batchProgress: number;
  currentSpeed: number;  // Records per second
  timeRemaining: number; // Seconds
  processingStats: {
    successCount: number;
    errorCount: number;
    lastBatchDuration: number;
    averageSpeed: number;
  };
  error?: string;
}>();

// Track connected WebSocket clients
const connectedClients = new Set<WebSocket>();

// Improved broadcastUpdate function
function broadcastUpdate(data: any) {
  if (data.type === 'progress') {
    try {
      // Handle Python service progress messages
      const progressStr = data.progress?.stage || '';
      const matches = progressStr.match(/(\d+)\/(\d+)/);
      const currentRecord = matches ? parseInt(matches[1]) : 0;
      const totalRecords = matches ? parseInt(matches[2]) : data.progress?.total || 0;
      const processedCount = data.progress?.processed || currentRecord;

      // Create enhanced progress object
      const enhancedProgress = {
        type: 'progress',
        progress: {
          processed: processedCount,
          total: totalRecords,
          stage: data.progress?.stage || 'Processing...',
          batchNumber: currentRecord,
          totalBatches: totalRecords,
          batchProgress: totalRecords > 0 ? Math.round((processedCount / totalRecords) * 100) : 0,
          currentSpeed: data.progress?.currentSpeed || 0,
          timeRemaining: data.progress?.timeRemaining || 0,
          processingStats: {
            successCount: processedCount,
            errorCount: data.progress?.processingStats?.errorCount || 0,
            averageSpeed: data.progress?.processingStats?.averageSpeed || 0
          }
        }
      };

      // Send to all connected clients
      const message = JSON.stringify(enhancedProgress);
      connectedClients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
          try {
            client.send(message);
          } catch (error) {
            console.error('Failed to send WebSocket message:', error);
          }
        }
      });
    } catch (error) {
      console.error('Error processing progress update:', error);
    }
  }
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

    // Send initial data
    storage.getSentimentPosts().then(posts => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'initial_data',
          data: posts
        }));
      }
    });

    // Handle client disconnection
    ws.on('close', () => {
      console.log('WebSocket client disconnected');
      connectedClients.delete(ws);
    });

    // Handle client messages
    ws.on('message', (message: string) => {
      try {
        const data = JSON.parse(message.toString());
        console.log('Received message:', data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    });
  });

  // Add the SSE endpoint inside registerRoutes
  app.get('/api/upload-progress/:sessionId', (req: Request, res: Response) => {
    const sessionId = req.params.sessionId;

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    });

    // Send initial progress data
    res.write(`data: ${JSON.stringify({
      processed: 0,
      total: 100,
      stage: "Initializing...",
      batchProgress: 0,
      currentSpeed: 0,
      timeRemaining: 0,
      processingStats: {
        successCount: 0,
        errorCount: 0,
        lastBatchDuration: 0,
        averageSpeed: 0
      }
    })}\n\n`);

    const sendProgress = () => {
      const progress = uploadProgressMap.get(sessionId);
      if (progress) {
        // Calculate real-time metrics
        const now = Date.now();
        const elapsed = (now - progress.timestamp) / 1000; // seconds

        if (elapsed > 0) {
          progress.currentSpeed = progress.processed / elapsed;
          progress.timeRemaining = progress.currentSpeed > 0 
            ? (progress.total - progress.processed) / progress.currentSpeed 
            : 0;
        }

        // Create enhanced progress object
        const enhancedProgress = {
          processed: progress.processed,
          total: progress.total || 100,
          stage: progress.stage || "Processing...",
          batchNumber: progress.batchNumber,
          totalBatches: progress.totalBatches,
          batchProgress: progress.batchProgress,
          currentSpeed: Math.round(progress.currentSpeed * 100) / 100,
          timeRemaining: Math.round(progress.timeRemaining),
          processingStats: progress.processingStats,
          error: progress.error
        };

        // Send to browser
        res.write(`data: ${JSON.stringify(enhancedProgress)}\n\n`);
      }
    };

    // Send progress immediately and then set interval
    sendProgress();
    const progressInterval = setInterval(sendProgress, 100);

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
      
      // Send all posts without filtering
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

  // Enhanced file upload endpoint
  app.post('/api/upload-csv', upload.single('file'), async (req: Request, res: Response) => {
    let sessionId = '';
    // Track the highest progress value to prevent jumping backward
    let highestProcessedValue = 0;

    // Log start of a new upload
    console.log('Starting new CSV upload, resetting progress tracking');

    let updateProgress = (
      processed: number, 
      stage: string, 
      total?: number,
      batchInfo?: {
        batchNumber: number;
        totalBatches: number;
        batchProgress: number;
        stats: {
          successCount: number;
          errorCount: number;
          lastBatchDuration: number;
          averageSpeed: number;
        };
      },
      error?: string
    ) => {
      if (sessionId) {
        // Log the raw progress update from Python service
        console.log('Raw progress update:', { processed, stage, total });

        // Get existing progress from the map
        const existingProgress = uploadProgressMap.get(sessionId);
        const existingTotal = existingProgress?.total || 0;

        // Try to extract progress data from PROGRESS: messages
        let extractedProcessed = processed;
        let extractedTotal = total || existingTotal;
        let extractedStage = stage;

        // Check if the stage message contains a JSON progress report
        if (stage.includes("PROGRESS:")) {
          try {
            // Extract the JSON portion from the PROGRESS: message
            const jsonStartIndex = stage.indexOf("PROGRESS:");
            const jsonString = stage.substring(jsonStartIndex + 9).trim();
            const progressJson = JSON.parse(jsonString);

            // Update with more accurate values from the progress message
            if (progressJson.processed !== undefined) {
              extractedProcessed = progressJson.processed;
            }
            if (progressJson.total !== undefined) {
              extractedTotal = progressJson.total;
            }
            if (progressJson.stage) {
              extractedStage = progressJson.stage;
            }

            console.log('Extracted progress from message:', { 
              extractedProcessed, 
              extractedStage, 
              extractedTotal 
            });
          } catch (err) {
            console.error('Failed to parse PROGRESS message:', err);
          }
        }

        // Handle "Completed record X/Y" format
        if (stage.includes("Completed record")) {
          const matches = stage.match(/Completed record (\d+)\/(\d+)/);
          if (matches) {
            extractedProcessed = parseInt(matches[1]);
            extractedTotal = parseInt(matches[2]);
            console.log('Extracted progress from completed record:', { 
              extractedProcessed, 
              extractedTotal 
            });
          }
        }

        // Prevent progress from going backward
        if (extractedProcessed < highestProcessedValue) {
          console.log(`Progress went backward (${extractedProcessed} < ${highestProcessedValue}), maintaining highest value`);
          extractedProcessed = highestProcessedValue;
        } else if (extractedProcessed > highestProcessedValue) {
          highestProcessedValue = extractedProcessed;
        }

        // Create progress update for broadcasting
        const progressData = {
          type: 'progress',
          sessionId,
          progress: {
            processed: extractedProcessed,
            total: extractedTotal,
            stage: extractedStage,
            timestamp: Date.now(),
            batchNumber: batchInfo?.batchNumber || 0,
            totalBatches: batchInfo?.totalBatches || 0,
            batchProgress: batchInfo?.batchProgress || 0,
            processingStats: batchInfo?.stats || {
              successCount: extractedProcessed,
              errorCount: 0,
              lastBatchDuration: 0,
              averageSpeed: 0
            }
          }
        };

        // Log the formatted progress data before broadcasting
        console.log('Formatted progress data:', progressData);

        broadcastUpdate(progressData);
      }
    };

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

      // Initialize enhanced progress tracking
      uploadProgressMap.set(sessionId, {
        processed: 0,
        total: totalRecords,
        stage: `Preparing to process ${totalRecords} records in batches of 30...`,
        timestamp: Date.now(),
        batchNumber: 0,
        totalBatches: Math.ceil(totalRecords / 30), 
        batchProgress: 0,
        currentSpeed: 0,
        timeRemaining: 0,
        processingStats: {
          successCount: 0,
          errorCount: 0,
          lastBatchDuration: 0,
          averageSpeed: 0
        }
      });

      // Send initial progress to connected clients
      broadcastUpdate({
        type: 'progress',
        sessionId,
        progress: uploadProgressMap.get(sessionId)
      });


      const { data, storedFilename, recordCount } = await pythonService.processCSV(
        fileBuffer,
        originalFilename,
        updateProgress
      );

      // Filter out non-disaster content using the same strict validation as real-time analysis
      const filteredResults = data.results.filter(post => {
        const isNonDisasterInput = post.text.length < 9 || 
                                  !post.explanation || 
                                  post.disasterType === "Not Specified" ||
                                  !post.disasterType ||
                                  post.text.match(/^[!?.,;:*\s]+$/);

        return !isNonDisasterInput;
      });

      // Log the filtering results
      console.log(`Filtered ${data.results.length - filteredResults.length} non-disaster posts out of ${data.results.length} total posts`, 'routes');

      // Save the analyzed file record
      const analyzedFile = await storage.createAnalyzedFile(
        insertAnalyzedFileSchema.parse({
          originalName: originalFilename,
          storedName: storedFilename,
          recordCount: filteredResults.length, 
          evaluationMetrics: data.metrics
        })
      );

      // Save only the filtered sentiment posts
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

      // Generate disaster events from the sentiment posts
      await generateDisasterEvents(sentimentPosts);

      // Final progress update
      if (sessionId && updateProgress) {
        updateProgress(totalRecords, 'Analysis complete', totalRecords);
      }

      // After successful processing, broadcast the new data
      broadcastUpdate({
        type: 'new_data',
        data: {
          posts: sentimentPosts,
          file: analyzedFile
        }
      });

      res.json({
        file: analyzedFile,
        posts: sentimentPosts,
        metrics: data.metrics,
        sessionId
      });
    } catch (error) {
      console.error("Error processing CSV:", error);

      updateProgress(0, 'Error', undefined, undefined, error instanceof Error ? error.message : String(error));

      res.status(500).json({ 
        error: "Failed to process CSV file",
        details: error instanceof Error ? error.message : String(error)
      });
    } finally {
      setTimeout(() => {
        uploadProgressMap.delete(sessionId);
      }, 5000);
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

  // Endpoint to get Python console messages
  // API endpoint to get daily usage stats
  app.get('/api/usage-stats', async (req: Request, res: Response) => {
    try {
      const stats = usageTracker.getUsageStats();
      res.json(stats);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch usage statistics" });
    }
  });

  app.get('/api/python-console-messages', async (req: Request, res: Response) => {
    try {
      // Return the most recent messages, with a limit of 100
      const limit = parseInt(req.query.limit as string) || 100;

      // Filter out noise and technical error messages that don't provide value to users
      const filteredMessages = pythonConsoleMessages.filter(item => {
        const message = item.message.toLowerCase();

        // Skip empty messages
        if (!item.message.trim()) return false;

        // Skip purely technical error messages with no user value
        if (
          (message.includes('traceback') && message.includes('error:')) ||
          message.includes('command failed with exit code') ||
          message.includes('deprecated') ||
          message.includes('warning: ') ||
          message.match(/^\s*at\s+[\w./<>]+:\d+:\d+\s*$/) // Stack trace lines
        ) {
          return false;
        }

        return true;
      });

      const recentMessages = filteredMessages
        .slice(-limit)
        .map(item => ({
          message: item.message,
          timestamp: item.timestamp.toISOString()
        }));

      res.json(recentMessages);
    } catch (error) {
      res.status(500).json({ 
        error: "Failed to retrieve Python console messages",
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

      res.send(csv);    } catch (error) {
      res.status(500).json({ error: "Failed to export CSV" });
    }
  });

  return httpServer;
}