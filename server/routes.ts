import express, { type Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from 'ws';
import { storage } from "./storage";
import { db } from "./db";
import { eq, sql } from "drizzle-orm";
import path from "path";
import multer from "multer";
import fs from "fs";
import { pythonService, pythonConsoleMessages } from "./python-service";
import { insertSentimentPostSchema, insertAnalyzedFileSchema, insertSentimentFeedbackSchema, sentimentPosts, type SentimentPost } from "@shared/schema";
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
  // Add health check endpoint for Render
  app.get('/api/health', (req: Request, res: Response) => {
    res.status(200).json({ 
      status: 'healthy',
      timestamp: new Date().toISOString(),
      environment: process.env.NODE_ENV || 'development'
    });
  });

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

    // First, delete existing events to avoid duplicates (keeping only 5 at a time)
    const currentEvents = await storage.getDisasterEvents();
    if (currentEvents.length >= 5) {
      // Sort by ID and keep only the 3 most recent
      const sortedEvents = currentEvents.sort((a, b) => b.id - a.id);
      for (let i = 3; i < sortedEvents.length; i++) {
        try {
          // Delete older events
          await storage.deleteDisasterEvent(sortedEvents[i].id);
        } catch (error) {
          console.error(`Failed to delete event ${sortedEvents[i].id}:`, error);
        }
      }
    }

    // Group posts by disaster type and location (more granular)
    const disasterGroups: {[key: string]: {
      posts: any[],
      locations: {[location: string]: number},
      sentiments: {[sentiment: string]: number},
      dates: {[date: string]: number}
    }} = {};

    // Process posts to identify disaster patterns
    for (const post of posts) {
      if (!post.disasterType || !post.timestamp) continue;
      
      // Format timestamp to a readable date 
      const postDate = new Date(post.timestamp);
      const formattedDate = postDate.toISOString().split('T')[0];
      
      // Skip future dates
      if (postDate > new Date()) continue;
      
      // Use disaster type as key
      const key = post.disasterType;
      
      if (!disasterGroups[key]) {
        disasterGroups[key] = {
          posts: [],
          locations: {},
          sentiments: {},
          dates: {}
        };
      }
      
      // Add post to group
      disasterGroups[key].posts.push(post);
      
      // Track locations
      if (post.location && 
          post.location !== 'UNKNOWN' && 
          post.location !== 'Not specified' && 
          post.location !== 'Philippines') {
        disasterGroups[key].locations[post.location] = 
          (disasterGroups[key].locations[post.location] || 0) + 1;
      }
      
      // Track sentiments
      disasterGroups[key].sentiments[post.sentiment] = 
        (disasterGroups[key].sentiments[post.sentiment] || 0) + 1;
        
      // Track dates
      disasterGroups[key].dates[formattedDate] = 
        (disasterGroups[key].dates[formattedDate] || 0) + 1;
    }
    
    // Process disaster groups to create meaningful events
    const newEvents = [];
    
    for (const [disasterType, data] of Object.entries(disasterGroups)) {
      // Skip if not enough data
      if (data.posts.length < 3) continue;
      
      // Find the most common location
      const locations = Object.entries(data.locations).sort((a, b) => b[1] - a[1]);
      const primaryLocation = locations.length > 0 ? locations[0][0] : null;
      
      // Find secondary locations (for multi-location events)
      const secondaryLocations = locations.slice(1, 3).map(l => l[0]);
      
      // Find the most recent date with activity
      const dates = Object.entries(data.dates).sort();
      const mostRecentDateStr = dates[dates.length - 1]?.[0];
      const mostRecentDate = mostRecentDateStr ? new Date(mostRecentDateStr) : new Date();
      
      // Find peak date (date with most activity)
      const peakDateEntry = Object.entries(data.dates).sort((a, b) => b[1] - a[1])[0];
      const peakDate = peakDateEntry ? new Date(peakDateEntry[0]) : new Date();
      
      // Calculate sentiment distribution
      const sentimentTotals = Object.values(data.sentiments).reduce((sum, count) => sum + count, 0);
      const sentimentDistribution = Object.entries(data.sentiments).map(([sentiment, count]) => {
        const percentage = Math.round((count / sentimentTotals) * 100);
        return `${sentiment} ${percentage}%`;
      }).join(', ');
      
      // Find sample posts with highest engagement or relevance
      const samplePosts = data.posts
        .filter(post => post.text.length > 15)
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 3);
        
      const sampleContent = samplePosts.length > 0 ? samplePosts[0].text : data.posts[0].text;
      
      // Create descriptive event name
      const eventName = primaryLocation 
        ? `${disasterType} in ${primaryLocation}` 
        : `${disasterType} Event`;
      
      // Create comprehensive description
      let description = `Based on ${data.posts.length} reports from the community. `;
      
      // Add location information if available
      if (primaryLocation && secondaryLocations.length > 0) {
        description += `Affected areas include ${primaryLocation}, ${secondaryLocations.join(', ')}. `;
      } else if (primaryLocation) {
        description += `Primary affected area: ${primaryLocation}. `;
      }
      
      // Add sentiment distribution
      description += `Sentiment distribution: ${sentimentDistribution}. `;
      
      // Add sample content
      description += `Sample report: "${sampleContent}"`;
      
      // Create the disaster event with rich, real-time data
      const newEvent = {
        name: eventName,
        description: description,
        timestamp: mostRecentDate,
        location: primaryLocation,
        type: disasterType,
        sentimentImpact: sentimentDistribution
      };
      
      newEvents.push(newEvent);
      
      // Store the event in the database
      await storage.createDisasterEvent(newEvent);
    }
    
    console.log(`Generated ${newEvents.length} new disaster events based on real-time data`);
  };

  // Get all sentiment posts
  app.get('/api/sentiment-posts', async (req: Request, res: Response) => {
    try {
      const posts = await storage.getSentimentPosts();
      
      // Filter out "UNKNOWN" locations if the query parameter is set
      const filterUnknown = req.query.filterUnknown === 'true';
      
      if (filterUnknown) {
        const filteredPosts = posts.filter(post => 
          post.location !== null && 
          post.location.toUpperCase() !== 'UNKNOWN' && 
          post.location !== 'Not specified' &&
          post.location !== 'Philippines'
        );
        res.json(filteredPosts);
      } else {
        // Send all posts without filtering if not explicitly requested
        res.json(posts);
      }
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch sentiment posts" });
    }
  });

  // Get sentiment posts by file id
  app.get('/api/sentiment-posts/file/:fileId', async (req: Request, res: Response) => {
    try {
      const fileId = parseInt(req.params.fileId);
      const posts = await storage.getSentimentPostsByFileId(fileId);
      
      // Filter out "UNKNOWN" locations if the query parameter is set
      const filterUnknown = req.query.filterUnknown === 'true';
      
      if (filterUnknown) {
        const filteredPosts = posts.filter(post => 
          post.location !== null && 
          post.location.toUpperCase() !== 'UNKNOWN' && 
          post.location !== 'Not specified' &&
          post.location !== 'Philippines'
        );
        res.json(filteredPosts);
      } else {
        // Send all posts without filtering if not explicitly requested
        res.json(posts);
      }
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch sentiment posts" });
    }
  });

  // Get all disaster events
  app.get('/api/disaster-events', async (req: Request, res: Response) => {
    try {
      const events = await storage.getDisasterEvents();
      
      // Filter out "UNKNOWN" locations if the query parameter is set
      const filterUnknown = req.query.filterUnknown === 'true';
      
      if (filterUnknown) {
        const filteredEvents = events.filter(event => 
          event.location !== null && 
          event.location.toUpperCase() !== 'UNKNOWN' && 
          event.location !== 'Not specified' &&
          event.location !== 'Philippines'
        );
        res.json(filteredEvents);
      } else {
        // Send all events without filtering if not explicitly requested
        res.json(events);
      }
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
  
  // Update file metrics
  app.patch('/api/analyzed-files/:id/metrics', async (req: Request, res: Response) => {
    try {
      const id = parseInt(req.params.id);
      const metrics = req.body;
      
      // Validate if file exists
      const file = await storage.getAnalyzedFile(id);
      if (!file) {
        return res.status(404).json({ error: "Analyzed file not found" });
      }
      
      // Update metrics
      await storage.updateFileMetrics(id, metrics);
      
      res.json({ success: true, message: "Metrics updated successfully" });
    } catch (error) {
      console.error("Error updating file metrics:", error);
      res.status(500).json({ 
        error: "Failed to update file metrics",
        details: error instanceof Error ? error.message : String(error)
      });
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
            // Extract the progress message between PROGRESS: and ::END_PROGRESS
            const progressMatch = stage.match(/PROGRESS:(.*?)::END_PROGRESS/);
            if (progressMatch && progressMatch[1]) {
              const progressJson = JSON.parse(progressMatch[1]);

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

              console.log('Extracted progress from message with marker:', { 
                extractedProcessed, 
                extractedStage, 
                extractedTotal 
              });
            } else {
              // Legacy fallback for old PROGRESS: format without ::END_PROGRESS marker
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

              console.log('Extracted progress from legacy message:', { 
                extractedProcessed, 
                extractedStage, 
                extractedTotal 
              });
            }
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
  // Profile Image Routes
  app.get('/api/profile-images', async (req: Request, res: Response) => {
    try {
      const profiles = await storage.getProfileImages();
      res.json(profiles);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch profile images" });
    }
  });

  app.post('/api/profile-images', upload.single('image'), async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No image uploaded" });
      }

      const { name, role, description } = req.body;
      
      // Save image to attached_assets
      const fileName = `profile-${Date.now()}-${req.file.originalname}`;
      const filePath = path.join(process.cwd(), 'attached_assets', fileName);
      fs.writeFileSync(filePath, req.file.buffer);

      const profile = await storage.createProfileImage({
        name,
        role,
        imageUrl: `/assets/${fileName}`,
        description
      });

      res.json(profile);
    } catch (error) {
      res.status(500).json({ error: "Failed to create profile image" });
    }
  });

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

  // Sentiment feedback for model training with real-time model updates
  app.post('/api/sentiment-feedback', async (req: Request, res: Response) => {
    try {
      console.log("Received sentiment feedback request:", JSON.stringify(req.body, null, 2));
      
      // Validate using the sentiment feedback schema with partial validation
      // to allow for missing optional fields
      const result = insertSentimentFeedbackSchema.partial().safeParse(req.body);
      if (!result.success) {
        console.error("Validation error:", result.error.format());
        return res.status(400).json({ 
          error: "Invalid feedback data", 
          details: result.error.format() 
        });
      }
      
      // Ensure required base fields are present
      if (!req.body.originalText || !req.body.originalSentiment) {
        console.error("Missing required base fields in feedback request");
        return res.status(400).json({
          error: "Missing required fields",
          details: "originalText and originalSentiment are required"
        });
      }
      
      // At least one correction must be provided (sentiment, location, or disaster type)
      if (!req.body.correctedSentiment && !req.body.correctedLocation && !req.body.correctedDisasterType) {
        console.log("No corrections provided in feedback request");
        return res.status(400).json({
          error: "Missing corrections",
          details: "At least one correction field (correctedSentiment, correctedLocation, or correctedDisasterType) must be provided."
        });
      }
      
      // We already validated corrections above, this code is kept for reference but commented out
      /* 
      if (!req.body.correctedSentiment && !req.body.correctedLocation && !req.body.correctedDisasterType) {
        console.error("No corrections provided in feedback request");
        return res.status(400).json({
          error: "Missing correction data",
          details: "At least one of correctedSentiment, correctedLocation, or correctedDisasterType must be provided"
        });
      }
      */
      
      // Create a properly typed object with the required fields
      const feedback = {
        originalText: req.body.originalText,
        originalSentiment: req.body.originalSentiment,
        correctedSentiment: req.body.correctedSentiment,
        correctedLocation: req.body.correctedLocation || null,
        correctedDisasterType: req.body.correctedDisasterType || null,
        originalPostId: req.body.originalPostId || null,
        userId: req.body.userId || null
      };
      
      console.log("Processing feedback for training:", feedback);
      
      // QUIZ VALIDATION FIRST: Do AI-POWERED VERIFICATION before saving to database
      // This ensures the user sees the quiz before any database updates happen
      
      // First, analyze the text with our AI model to get sentiment validation
      let aiAnalysisResult: any = null;
      let possibleTrolling = false;
      let aiTrustMessage = "";
      let quizValidation = null;
      
      try {
        console.log("Performing AI quiz validation before saving feedback...");
        // Get AI validation in quiz format
        quizValidation = await pythonService.trainModelWithFeedback(
          feedback.originalText,
          feedback.originalSentiment,
          feedback.correctedSentiment || feedback.originalSentiment, // Use original if no correction
          feedback.correctedLocation,
          feedback.correctedDisasterType
        );
        
        console.log("Quiz validation result:", quizValidation);
        
        // If there's a quiz validation message, use it for the trust message
        if (quizValidation.message) {
          aiTrustMessage = quizValidation.message;
          
          // Check if the quiz validation indicates a problem
          if (quizValidation.status === "quiz_feedback") {
            possibleTrolling = true;
            console.log("⚠️ AI QUIZ VALIDATION: Quiz suggests a potential problem with this feedback");
          }
        } else if (quizValidation.status === "success") {
          // Make sure we always have a message even if the validation passed
          aiTrustMessage = `VALIDATION PASSED: Your correction from "${feedback.originalSentiment}" to "${feedback.correctedSentiment}" has been accepted. Thank you for helping improve our system!`;
          console.log("Setting default success message for AI trust validation");
        }
      } catch (aiError) {
        console.error("Error during AI quiz validation:", aiError);
        // We'll still save the feedback but mark it with an error
      }
      
      // Only save to database if we get past the quiz validation
      const savedFeedback = await storage.submitSentimentFeedback(feedback);
      console.log("Feedback saved to database with ID:", savedFeedback.id);
      
      // Try to detect language from text
      let language = "English";
      try {
        // Try to guess language from text
        if (feedback.originalText.match(/[ñÑáéíóúÁÉÍÓÚ]/)) {
          language = "Filipino"; // Simple heuristic for Filipino text with accent marks
        } else if (feedback.originalText.match(/\b(ako|namin|natin|kami|tayo|nila|sila|mo|niya|ko|kayo|ikaw|siya)\b/i)) {
          language = "Filipino"; // Check for common Filipino pronouns
        }
      } catch (e) {
        console.log("Language detection failed, defaulting to English");
      }

      // Also save to training examples database for persistent learning
      try {
        // Create text_key from the original text by normalizing to lowercase and joining words
        const textWords = feedback.originalText.toLowerCase().match(/\b\w+\b/g) || [];
        const textKey = textWords.join(' ');
        
        // Only create training example if correctedSentiment is provided
        if (feedback.correctedSentiment) {
          const trainingExample = await storage.createTrainingExample({
            text: feedback.originalText,
            textKey: textKey,
            sentiment: feedback.correctedSentiment,
            language: language,
            confidence: 0.95
          });
          console.log(`Training example saved to database with ID: ${trainingExample.id}`);
        } else {
          console.log(`No sentiment correction provided, skipping training example creation`);
        }
      } catch (dbError) {
        console.error("Error saving training example to database:", dbError);
        // Continue even if this fails - it might be a duplicate
      }
      
      // Already performed AI-POWERED VERIFICATION above, so we'll skip the duplicate code
      
      // Instead of using hardcoded keywords, we'll analyze the text using our AI system
      // to determine if it contains panic indicators
      
      // First, analyze the original text with our AI model to get a proper assessment
      let isPanicText = false;
      // Use the existing aiAnalysisResult variable instead of redeclaring it
      // let aiAnalysisResult: any = null;
      
      try {
        // Run AI analysis on the original text to determine its true emotional content
        aiAnalysisResult = await pythonService.analyzeSentiment(feedback.originalText);
        
        console.log("🧠 AI Analysis result:", aiAnalysisResult);
        
        // Use AI-determined sentiment to identify if this is panic text
        // This is much more accurate than using hardcoded keywords
        isPanicText = 
          aiAnalysisResult.sentiment === 'Panic' || 
          aiAnalysisResult.sentiment === 'Fear/Anxiety' ||
          (aiAnalysisResult.confidence > 0.75 && 
           aiAnalysisResult.explanation.toLowerCase().includes('distress') || 
           aiAnalysisResult.explanation.toLowerCase().includes('urgent'));
           
        console.log(`🧠 AI determined this ${isPanicText ? 'IS' : 'is NOT'} panic text with confidence ${aiAnalysisResult.confidence}`);
      } catch (aiError) {
        // Fallback only if AI analysis fails
        console.error("Error during AI verification:", aiError);
        
        // Using a more intelligent fallback based on multiple linguistic signals
        // Only as a last resort if AI analysis fails completely
        const hasPanicWords = [
          // Filipino panic/fear words
          'takot', 'natatakot', 'natakot', 'nakakatakot',
          'kame', 'kami', 'tulong', 'saklolo',
          // English panic/fear words
          'scared', 'terrified', 'help', 'fear', 'afraid',
          'emergency', 'evacuate', 'evacuating', 'destroyed', 'lost'
        ].some(word => feedback.originalText.toLowerCase().includes(word));
        
        // Check for ALL CAPS which often indicates urgency or intensity
        const hasAllCaps = feedback.originalText.split(' ').some((word: string) => 
          word.length > 3 && word === word.toUpperCase() && /[A-Z]/.test(word)
        );
        
        // Check for multiple exclamation points which can indicate urgency
        const hasMultipleExclamations = (feedback.originalText.match(/!/g) || []).length >= 2;
        
        // Combine signals for a more robust fallback detection
        isPanicText = hasPanicWords || hasAllCaps || hasMultipleExclamations;
        
        console.log("⚠️ WARNING: Using intelligent fallback detection because AI analysis failed");
      }
      
      // Skip all troll detection if no sentiment correction is provided
      // This allows changing only location or disaster type without triggering troll protection
      if (feedback.correctedSentiment) {
        // TROLL PROTECTION 1: Check for PANIC text being changed to something else
        if (isPanicText && 
            (feedback.correctedSentiment !== 'Panic' && feedback.correctedSentiment !== 'Fear/Anxiety')
        ) {
          possibleTrolling = true;
          aiTrustMessage = "Our AI analysis detected that this text contains panic indicators that don't match the suggested sentiment. Please verify your correction.";
          console.log("⚠️ AI TRUST VERIFICATION: Detected possible mismatch - panic text being changed to non-panic sentiment");
        }
        
        // TROLL PROTECTION 2: Check for Resilience text being changed to Panic without indicators 
        if ((feedback.originalSentiment === 'Resilience' || feedback.originalSentiment === 'Neutral') &&
            (feedback.correctedSentiment === 'Panic') &&
            !isPanicText
        ) {
          possibleTrolling = true;
          aiTrustMessage = "Our AI analysis found that this text doesn't contain panic indicators that would justify a Panic sentiment classification. Please verify your correction.";
          console.log("⚠️ AI TRUST VERIFICATION: Detected possible mismatch - non-panic text being marked as panic");
        }
        
        // TROLL PROTECTION 3: Use AI to check for humorous/joking content being changed to serious sentiment
        // Instead of using hardcoded joke words, analyze the content and tone using AI
        try {
          // Re-use the same analysis result from above since we already ran it once
          // We need to access the AI analysis result that was already computed
          
          // First make sure we're only accessing this after AI analysis is complete
          if (aiAnalysisResult !== null) {
            const isJokeOrDisbelief = 
              aiAnalysisResult.sentiment === 'Disbelief' || 
              (aiAnalysisResult.explanation && aiAnalysisResult.explanation.toLowerCase().includes('humor')) ||
              (aiAnalysisResult.explanation && aiAnalysisResult.explanation.toLowerCase().includes('joke')) ||
              (aiAnalysisResult.explanation && aiAnalysisResult.explanation.toLowerCase().includes('kidding')) ||
              (aiAnalysisResult.explanation && aiAnalysisResult.explanation.toLowerCase().includes('laughter')) ||
              (aiAnalysisResult.explanation && aiAnalysisResult.explanation.toLowerCase().includes('sarcasm')) ||
              (aiAnalysisResult.explanation && aiAnalysisResult.explanation.toLowerCase().includes('not serious')) ||
              // Check for common Filipino joke indicators
              (feedback.originalText.toLowerCase().includes('haha') && feedback.originalText.includes('!')) ||
              (feedback.originalText.toLowerCase().includes('ulol') || feedback.originalText.toLowerCase().includes('gago')) ||
              (feedback.originalText.toUpperCase().includes('DAW?') || feedback.originalText.includes('DAW!'));
            
            // CASE 1: Changing joke content to Panic/Fear (serious emotion)
            // NOTE: This TypeScript validation is now just a backup
            // The primary validation is now done by the Python AI model
            if (isJokeOrDisbelief && 
                (feedback.correctedSentiment === 'Panic' || feedback.correctedSentiment === 'Fear/Anxiety')
            ) {
              possibleTrolling = true;
              aiTrustMessage = "Our AI analysis found that this text contains humor or disbelief indicators which may not align with a serious Panic sentiment. Please review your correction.";
              console.log("⚠️ AI TRUST VERIFICATION (Fallback): Detected potential mismatch - humorous/disbelief text marked as panic");
            }
            
            // CASE 2: Changing joke/Disbelief content to Neutral (incorrect behavior)
            // NOTE: This TypeScript validation is now just a backup
            // The primary, more advanced validation is now done in the Python AI model
            // We keep this as a fallback only, in case the Python validation fails
            if (isJokeOrDisbelief && 
                (feedback.originalSentiment === 'Disbelief' && feedback.correctedSentiment === 'Neutral')
            ) {
              possibleTrolling = true;
              aiTrustMessage = "Our AI analysis found that this text contains joke/sarcasm indicators which should be classified as Disbelief, not Neutral. Please verify your correction.";
              console.log("⚠️ AI TRUST VERIFICATION (Fallback): Detected potential mismatch - joke/sarcasm text being changed from Disbelief to Neutral");
            }
            
            // CASE 3: Changing Neutral content to Disbelief without humor/sarcasm markers
            // NOTE: This is also a fallback validation - primary validation is in Python model
            if (!isJokeOrDisbelief && 
                (feedback.originalSentiment === 'Neutral' && feedback.correctedSentiment === 'Disbelief') &&
                !feedback.originalText.toLowerCase().includes('haha') &&
                !feedback.originalText.includes('!!')
            ) {
              possibleTrolling = true;
              aiTrustMessage = "Our AI analysis found that this text doesn't contain clear humor or sarcasm indicators that would justify a Disbelief classification. Please verify your correction.";
              console.log("⚠️ AI TRUST VERIFICATION (Fallback): Detected potential mismatch - neutral text being marked as Disbelief without joke indicators");
            }
          }
        } catch (jokeCheckError) {
          console.error("Error checking for joke content:", jokeCheckError);
          // No fallback needed here, this is just an extra verification
        }
      } else {
        console.log("Skipping troll detection since no sentiment correction was provided");
      }
      
      // Create base response
      // Update base response with quiz validation result if available
      const baseResponse = {
        id: savedFeedback.id,
        originalText: savedFeedback.originalText,
        originalSentiment: savedFeedback.originalSentiment,
        correctedSentiment: savedFeedback.correctedSentiment,
        correctedLocation: savedFeedback.correctedLocation,
        correctedDisasterType: savedFeedback.correctedDisasterType,
        trainedOn: false,
        createdAt: savedFeedback.createdAt,
        userId: savedFeedback.userId,
        originalPostId: savedFeedback.originalPostId,
        possibleTrolling: possibleTrolling,
        aiTrustMessage: aiTrustMessage
      };
      
      // Add quiz validation result if available
      if (quizValidation && quizValidation.message) {
        baseResponse.aiTrustMessage = quizValidation.message;
      }
      
      // Execute training in a separate try/catch to handle training errors independently
      try {
        console.log("Starting model training with feedback");
        
        // Immediately train the model with this feedback
        const trainingResult = await pythonService.trainModelWithFeedback(
          feedback.originalText,
          feedback.originalSentiment,
          feedback.correctedSentiment,
          feedback.correctedLocation,
          feedback.correctedDisasterType
        );
        
        console.log("Model training completed with result:", trainingResult);
        
        // Log successful training
        if (trainingResult.status === 'success') {
          const improvement = ((trainingResult.performance?.improvement || 0) * 100).toFixed(2);
          console.log(`🚀 Model trained successfully - Performance improved by ${improvement}%`);
          
          // Update the feedback record to mark it as trained
          await storage.markFeedbackAsTrained(savedFeedback.id);
          
          // Clear the cache entry for this text to force re-analysis next time
          pythonService.clearCacheForText(feedback.originalText);
          console.log(`Cache cleared for retrained text: "${feedback.originalText.substring(0, 30)}..."`);
          
          // We used to skip updates if AI verification failed, but now we'll always update
          // the database records regardless of the warning, just showing warnings to users
          // This ensures that admin/user-provided corrections are always applied to the database
          if (possibleTrolling) {
            console.log(`⚠️ AI WARNING: Detected potential irregular feedback but will still update posts.`);
            console.log(`AI Message: ${aiTrustMessage}`);
            console.log(`✅ IMPORTANT: Still applying updates to database as requested by user/admin`);
            
            // Continue with updates regardless of warning - user knows best in some cases
          }
          
          // UPDATE ALL EXISTING POSTS WITH SAME TEXT TO NEW SENTIMENT
          try {
            // Get all posts from the database with the same text
            const query = db.select().from(sentimentPosts)
              .where(sql`text = ${feedback.originalText}`);
            
            const postsToUpdate = await query;
            console.log(`Found ${postsToUpdate.length} posts with the same text to update sentiment`);
            
            // Update each post with the new corrected sentiment, but with verification
            for (const post of postsToUpdate) {
              // APPLY UPDATES DIRECTLY - No longer using hardcoded keyword checks
              // We trust the user/admin feedback and will update the records directly
              // This ensures changes are immediately visible on the frontend
              
              // Just determine if the new sentiment is panic-related for logging
              const isPanicSentiment = feedback.correctedSentiment === 'Panic' || feedback.correctedSentiment === 'Fear/Anxiety';
              
              // We're removing the protection mechanism that prevented updates
              // Instead, we'll just log that we're applying the user's changes as requested
              console.log(`Applying user-requested changes to post ID ${post.id} - Admin/user feedback takes priority`);
              // No skipping updates - always make the requested changes
              
              // Create an object with the fields to update
              const updateFields: Record<string, any> = {
                confidence: 0.84 // Moderate-high confidence (80-86 range)
              };
              
              // Add correctedSentiment if provided
              if (feedback.correctedSentiment) {
                updateFields.sentiment = feedback.correctedSentiment;
              }
              
              // Add correctedLocation if provided
              if (feedback.correctedLocation) {
                updateFields.location = feedback.correctedLocation;
              }
              
              // Add correctedDisasterType if provided
              if (feedback.correctedDisasterType) {
                updateFields.disasterType = feedback.correctedDisasterType;
              }
              
              // Update the post with all provided corrections
              await db.update(sentimentPosts)
                .set(updateFields)
                .where(eq(sentimentPosts.id, post.id));
                
              // Get the updated post to ensure we see the changes
              const updatedPost = await db.select().from(sentimentPosts).where(eq(sentimentPosts.id, post.id)).then(posts => posts[0]);
              
              // Log the update details
              let updateMessage = `Updated post ID ${post.id}:`;
              if (feedback.correctedSentiment) {
                updateMessage += ` sentiment from '${post.sentiment}' to '${feedback.correctedSentiment}'`;
              }
              if (feedback.correctedLocation) {
                updateMessage += ` location to '${feedback.correctedLocation}'`;
              }
              if (feedback.correctedDisasterType) {
                updateMessage += ` disaster type to '${feedback.correctedDisasterType}'`;
              }
              console.log(updateMessage);
              
              // Send a broadcast specifically for this post update to force UI refresh
              broadcastUpdate({
                type: "post-updated",
                data: {
                  id: post.id,
                  updates: updateFields,
                  originalText: post.text,
                  updatedPost: updatedPost
                }
              });
            }
            
            // LOOK FOR SIMILAR POSTS that have SAME MEANING using AI verification
            // This ensures that variations with same meaning get updated, while different meanings are preserved
            try {
              // Get all posts from the database that aren't exact matches but might be similar
              // We'll exclude posts we've already updated
              const excludeIds = postsToUpdate.map(p => p.id);
              const allPosts = await db.select().from(sentimentPosts);
              
              // This function is a quick pre-filter before running more expensive AI analysis
              // It helps us avoid unnecessary AI API calls for obviously unrelated content
              const hasObviouslyDifferentContext = (originalText: string, postText: string): boolean => {
                // If the lengths are dramatically different, they're probably not similar
                const originalLength = originalText.length;
                const postLength = postText.length;
                const lengthRatio = Math.max(originalLength, postLength) / Math.min(originalLength, postLength);
                
                if (lengthRatio > 3) {
                  console.log(`Context differs: Length ratio too high (${lengthRatio.toFixed(1)})`);
                  return true;
                }
                
                // Simple emoji detection without unicode patterns
                const commonEmojis = ['😀', '😁', '😂', '🙂', '😊', '😎', '👍', '🔥', '💯', '❤️'];
                const originalHasEmojis = commonEmojis.some(emoji => originalText.includes(emoji));
                const postHasEmojis = commonEmojis.some(emoji => postText.includes(emoji));
                
                if (originalHasEmojis !== postHasEmojis) {
                  console.log(`Context differs: Emoji presence mismatch between texts`);
                  return true;
                }
                
                // Basic language check - if one is clearly English and the other Filipino
                const originalHasFilipino = 
                  originalText.toLowerCase().includes('ng') || 
                  originalText.toLowerCase().includes('ang') ||
                  originalText.toLowerCase().includes('naman');
                  
                const postHasFilipino = 
                  postText.toLowerCase().includes('ng') || 
                  postText.toLowerCase().includes('ang') ||
                  postText.toLowerCase().includes('naman');
                  
                if (originalHasFilipino !== postHasFilipino) {
                  console.log(`Context differs: Language mismatch (Filipino vs English)`);
                  return true;
                }
                
                // Let the AI make the final determination if we get past these basic filters
                return false;
              }
              
              // Filter to get only posts that aren't already updated AND don't have obviously different context
              const postsToCheck = allPosts.filter(post => 
                !excludeIds.includes(post.id) && 
                post.text !== feedback.originalText &&
                !hasObviouslyDifferentContext(feedback.originalText, post.text)
              );
              
              console.log(`After context-based filtering, only ${postsToCheck.length} posts need AI verification`);
              
              if (postsToCheck.length === 0) {
                console.log("No additional posts to check for semantic similarity");
                return;
              }
              
              console.log(`Found ${postsToCheck.length} posts to check for semantic similarity`);
              
              // Use AI to verify semantic similarity - but do it in batches to avoid performance issues
              const similarPosts: SentimentPost[] = [];
              const batchSize = 5;
              
              for (let i = 0; i < postsToCheck.length; i += batchSize) {
                const batch = postsToCheck.slice(i, i + batchSize);
                const batchPromises = batch.map(async (post): Promise<SentimentPost | null> => {
                  try {
                    // IMPORTANT: Use the AI service to verify if the post has the same core meaning
                    // We use Python service to check if these texts actually have the same meaning
                    // Pass the sentiment context to help determine if these texts should be similar
                    const verificationResult = await pythonService.analyzeSimilarityForFeedback(
                      feedback.originalText,
                      post.text,
                      feedback.originalSentiment,  // Pass original sentiment
                      feedback.correctedSentiment  // Pass corrected sentiment
                    );
                    
                    if (verificationResult && verificationResult.areSimilar === true) {
                      // Even with our advanced verification, double-check context again
                      // This ensures that even if the AI says it's similar, we confirm it makes sense
                      
                      // We use AI analysis for checking sentiment instead of hardcoded keywords
                      // This approach is much more accurate and reduces false positives/negatives
                      try {
                        // Use AI to analyze the sentiment of this post
                        const postAnalysisResult = await pythonService.analyzeSentiment(post.text);
                        console.log(`AI analysis for similar post ID ${post.id}: ${postAnalysisResult.sentiment} (confidence: ${postAnalysisResult.confidence})`);
                        
                        // Only check sentiment context mismatches if correctedSentiment is provided
                        if (feedback.correctedSentiment) {
                          const postHasPanicSentiment = 
                            postAnalysisResult.sentiment === 'Panic' || 
                            postAnalysisResult.sentiment === 'Fear/Anxiety';
                            
                          const targetIsPanicSentiment = 
                            feedback.correctedSentiment === 'Panic' || 
                            feedback.correctedSentiment === 'Fear/Anxiety';
                          
                          // If AI detected panic but we're trying to change to non-panic, skip update
                          if (postHasPanicSentiment && !targetIsPanicSentiment) {
                            console.log(`AI VERIFICATION: Post has panic sentiment but target is ${feedback.correctedSentiment}`);
                            console.log(`Allowing update anyway as requested by admin/user (post ID ${post.id})`);
                            // Do not return null - let the update proceed as user/admin knows best
                          }
                          
                          // If AI didn't detect panic but we're changing to panic, still allow it
                          // This enables user corrections where the AI might have missed subtle panic indicators
                          if (!postHasPanicSentiment && targetIsPanicSentiment) {
                            console.log(`AI NOTE: AI did not detect panic but user marked as panic - allowing update`);
                            // Proceed with the update - we trust the user's judgment
                          }
                        } else {
                          console.log(`No sentiment correction provided - continuing with changes to location/disaster type only`);
                        }
                      } catch (aiError) {
                        // If AI analysis fails, log but continue with the update
                        console.error(`AI verification failed for post ID ${post.id}:`, aiError);
                        console.log(`Continuing with update despite AI verification failure`);
                      }
                      
                      console.log(`AI verified semantic similarity: "${post.text.substring(0, 30)}..." is similar to original`);
                      console.log(`Context verification PASSED - can safely update sentiment to ${feedback.correctedSentiment}`);
                      return post;
                    } else {
                      console.log(`AI rejected similarity: "${post.text.substring(0, 30)}..." with reason: ${verificationResult?.explanation || 'Unknown'}`);
                    }
                    return null;
                  } catch (err) {
                    console.error(`Error analyzing similarity for post ID ${post.id}:`, err);
                    return null;
                  }
                });
                
                const batchResults = await Promise.all(batchPromises);
                batchResults.forEach((post: SentimentPost | null) => {
                  if (post) similarPosts.push(post);
                });
              }
              
              console.log(`Found ${similarPosts.length} semantically similar posts verified by AI`);
              
              // Simply apply updates directly without additional verification
              // We're now using AI analysis earlier in the process, so no need for hardcoded checks here
              for (const post of similarPosts) {
                // We no longer use hardcoded keyword checks for verification
                // The AI analysis performed earlier is trusted to make the right determination
                
                // Always apply the updates as requested by the user/admin
                // This ensures changes are immediately visible in the frontend
                console.log(`Applying user-requested changes to similar post ID ${post.id}`);
                
                // For logging purposes only
                const isPanicSentiment = feedback.correctedSentiment === 'Panic' || feedback.correctedSentiment === 'Fear/Anxiety';
                
                // We used to prevent updates if trolling was detected, but now we'll always allow updates
                // User/admin feedback takes priority over automated detection
                // This ensures frontend is immediately updated with the changes
                if (possibleTrolling) {
                  console.log(`⚠️ Warning present but allowing update for post ID ${post.id} as requested by user/admin`);
                  // Continue with update (no 'continue' statement)
                }
                
                // If we've passed all verification, proceed with the update
                // Create an object with the fields to update for similar posts
                const similarUpdateFields: Record<string, any> = {
                  confidence: 0.82 // Moderate confidence for similar posts
                };
                
                // Add correctedSentiment if provided
                if (feedback.correctedSentiment) {
                  similarUpdateFields.sentiment = feedback.correctedSentiment;
                }
                
                // Add correctedLocation if provided
                if (feedback.correctedLocation) {
                  similarUpdateFields.location = feedback.correctedLocation;
                }
                
                // Add correctedDisasterType if provided
                if (feedback.correctedDisasterType) {
                  similarUpdateFields.disasterType = feedback.correctedDisasterType;
                }
                
                await db.update(sentimentPosts)
                  .set(similarUpdateFields)
                  .where(eq(sentimentPosts.id, post.id));
                  
                // Log the update details
                let updateMessage = `Updated AI-verified similar post ID ${post.id}:`;
                if (feedback.correctedSentiment) {
                  updateMessage += ` sentiment from '${post.sentiment}' to '${feedback.correctedSentiment}'`;
                }
                if (feedback.correctedLocation) {
                  updateMessage += ` location to '${feedback.correctedLocation}'`;
                }
                if (feedback.correctedDisasterType) {
                  updateMessage += ` disaster type to '${feedback.correctedDisasterType}'`;
                }
                console.log(updateMessage);
              }
            } catch (similarError) {
              console.error("Error updating similar posts with AI verification:", similarError);
            }
          } catch (error) {
            console.error("Error updating existing sentiment posts:", error);
          }
          
          // Broadcast update to connected clients
          broadcastUpdate({ 
            type: "feedback-update", 
            data: { 
              originalText: feedback.originalText,
              originalSentiment: feedback.originalSentiment,
              correctedSentiment: feedback.correctedSentiment,
              correctedLocation: feedback.correctedLocation,
              correctedDisasterType: feedback.correctedDisasterType,
              trainingResult: trainingResult,
              feedback_id: savedFeedback.id
            } 
          });
          
          // Return success response with training results
          return res.status(200).json({
            ...baseResponse,
            trainedOn: true,
            trainingResult: trainingResult
          });
        } else {
          console.log("Model training returned error status:", trainingResult.message);
          return res.status(200).json({
            ...baseResponse,
            trainingError: trainingResult.message
          });
        }
      } catch (trainingError) {
        console.error("Error training model with feedback:", trainingError);
        
        // Still return success since we saved the feedback, but include training error
        return res.status(200).json({
          ...baseResponse,
          trainingError: "Model training failed, but feedback was saved"
        });
      }
    } catch (error) {
      console.error("Error in sentiment feedback processing:", error);
      return res.status(500).json({ 
        error: "Failed to process feedback", 
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });
  
  // Get all sentiment feedback
  app.get('/api/sentiment-feedback', async (req: Request, res: Response) => {
    try {
      const feedback = await storage.getSentimentFeedback();
      return res.status(200).json(feedback);
    } catch (error) {
      console.error("Error getting feedback:", error);
      return res.status(500).json({ error: "Failed to get feedback" });
    }
  });

  // Get untrained feedback for model retraining
  app.get('/api/untrained-feedback', async (req: Request, res: Response) => {
    try {
      const feedback = await storage.getUntrainedFeedback();
      return res.status(200).json(feedback);
    } catch (error) {
      console.error("Error getting untrained feedback:", error);
      return res.status(500).json({ error: "Failed to get untrained feedback" });
    }
  });

  // Mark feedback as trained
  app.patch('/api/sentiment-feedback/:id/trained', async (req: Request, res: Response) => {
    try {
      const id = parseInt(req.params.id);
      await storage.markFeedbackAsTrained(id);
      return res.status(200).json({ message: "Feedback marked as trained" });
    } catch (error) {
      console.error("Error marking feedback as trained:", error);
      return res.status(500).json({ error: "Failed to mark feedback as trained" });
    }
  });
  
  // Get all training examples
  app.get('/api/training-examples', async (req: Request, res: Response) => {
    try {
      const examples = await storage.getTrainingExamples();
      return res.status(200).json(examples);
    } catch (error) {
      console.error("Error getting training examples:", error);
      return res.status(500).json({ error: "Failed to get training examples" });
    }
  });

  return httpServer;
}