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
import { insertSentimentPostSchema, insertAnalyzedFileSchema, insertSentimentFeedbackSchema, sentimentPosts } from "@shared/schema";
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
      
      // Ensure required fields are present
      if (!req.body.originalText || !req.body.originalSentiment || !req.body.correctedSentiment) {
        console.error("Missing required fields in feedback request");
        return res.status(400).json({
          error: "Missing required fields",
          details: "originalText, originalSentiment, and correctedSentiment are required"
        });
      }
      
      // Create a properly typed object with the required fields
      const feedback = {
        originalText: req.body.originalText,
        originalSentiment: req.body.originalSentiment,
        correctedSentiment: req.body.correctedSentiment,
        originalPostId: req.body.originalPostId || null,
        userId: req.body.userId || null
      };
      
      console.log("Processing feedback for training:", feedback);
      
      // Save to database first
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
        
        // Create training example from feedback
        const trainingExample = await storage.createTrainingExample({
          text: feedback.originalText,
          textKey: textKey,
          sentiment: feedback.correctedSentiment,
          language: language,
          confidence: 0.95
        });
        
        console.log(`Training example saved to database with ID: ${trainingExample.id}`);
      } catch (dbError) {
        console.error("Error saving training example to database:", dbError);
        // Continue even if this fails - it might be a duplicate
      }
      
      // Create base response
      const baseResponse = {
        id: savedFeedback.id,
        originalText: savedFeedback.originalText,
        originalSentiment: savedFeedback.originalSentiment,
        correctedSentiment: savedFeedback.correctedSentiment,
        trainedOn: false,
        createdAt: savedFeedback.createdAt,
        userId: savedFeedback.userId,
        originalPostId: savedFeedback.originalPostId
      };
      
      // Execute training in a separate try/catch to handle training errors independently
      try {
        console.log("Starting model training with feedback");
        
        // Immediately train the model with this feedback
        const trainingResult = await pythonService.trainModelWithFeedback(
          feedback.originalText,
          feedback.originalSentiment,
          feedback.correctedSentiment
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
          
          // UPDATE ALL EXISTING POSTS WITH SAME TEXT TO NEW SENTIMENT
          try {
            // Get all posts from the database with the same text
            const query = db.select().from(sentimentPosts)
              .where(sql`text = ${feedback.originalText}`);
            
            const postsToUpdate = await query;
            console.log(`Found ${postsToUpdate.length} posts with the same text to update sentiment`);
            
            // Update each post with the new corrected sentiment
            for (const post of postsToUpdate) {
              await db.update(sentimentPosts)
                .set({ 
                  sentiment: feedback.correctedSentiment, 
                  confidence: 0.95 // High confidence since this is manually corrected
                })
                .where(eq(sentimentPosts.id, post.id));
                
              console.log(`Updated post ID ${post.id} sentiment from ${post.sentiment} to ${feedback.correctedSentiment}`);
            }
            
            // LOOK FOR SIMILAR POSTS that have SAME MEANING using AI verification
            // This ensures that variations with same meaning get updated, while different meanings are preserved
            try {
              // Get all posts from the database that aren't exact matches but might be similar
              // We'll exclude posts we've already updated
              const excludeIds = postsToUpdate.map(p => p.id);
              const allPosts = await db.select().from(sentimentPosts);
              
              // Filter to get only posts that aren't already updated
              const postsToCheck = allPosts.filter(post => 
                !excludeIds.includes(post.id) && 
                post.text !== feedback.originalText &&
                post.sentiment !== feedback.correctedSentiment // Only consider posts with different sentiment
              );
              
              if (postsToCheck.length === 0) {
                console.log("No additional posts to check for semantic similarity");
                return;
              }
              
              console.log(`Found ${postsToCheck.length} posts to check for semantic similarity`);
              
              // Use AI to verify semantic similarity - but do it in batches to avoid performance issues
              const similarPosts = [];
              const batchSize = 5;
              
              for (let i = 0; i < postsToCheck.length; i += batchSize) {
                const batch = postsToCheck.slice(i, i + batchSize);
                const batchPromises = batch.map(async (post) => {
                  try {
                    // IMPORTANT: Use the AI service to verify if the post has the same core meaning
                    // We use Python service to check if these texts actually have the same meaning
                    const verificationResult = await pythonService.analyzeSimilarityForFeedback(
                      feedback.originalText,
                      post.text
                    );
                    
                    if (verificationResult && verificationResult.areSimilar === true) {
                      console.log(`AI verified semantic similarity: "${post.text.substring(0, 30)}..." is similar to original`);
                      return post;
                    }
                    return null;
                  } catch (err) {
                    console.error(`Error analyzing similarity for post ID ${post.id}:`, err);
                    return null;
                  }
                });
                
                const batchResults = await Promise.all(batchPromises);
                batchResults.forEach(post => {
                  if (post) similarPosts.push(post);
                });
              }
              
              console.log(`Found ${similarPosts.length} semantically similar posts verified by AI`);
              
              // Update the truly similar posts
              for (const post of similarPosts) {
                await db.update(sentimentPosts)
                  .set({ 
                    sentiment: feedback.correctedSentiment, 
                    confidence: 0.92 // High confidence based on verified similarity
                  })
                  .where(eq(sentimentPosts.id, post.id));
                  
                console.log(`Updated AI-verified similar post ID ${post.id} sentiment from ${post.sentiment} to ${feedback.correctedSentiment}`);
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