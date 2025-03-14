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

// Enhanced emotion detection with more sophisticated patterns
const emotionPatterns = {
  Panic: {
    keywords: [
      'panic', 'terrified', 'horrified', 'scared', 'frightened', 'takot', 'natatakot',
      'emergency', 'help', 'tulong', 'evacuate', 'evacuating', 'run', 'flee', 'escape',
      'trapped', 'stuck', 'nasukol', 'naiipit', 'hindi makalabas', 'can\'t get out',
      'SOS', 'mayday', 'danger', 'dangerous', 'delikado', 'mapanganib'
    ],
    intensifiers: ['very', 'really', 'extremely', 'sobra', 'grabe', 'napaka', 'super'],
    contextual: ['need immediate', 'right now', 'quickly', 'urgent', 'emergency'],
    weight: 2.0
  },
  'Fear/Anxiety': {
    keywords: [
      'fear', 'worried', 'anxious', 'nervous', 'kabado', 'nag-aalala', 'balisa',
      'concerned', 'scared', 'afraid', 'natatakot', 'kinakabahan', 'nangangamba',
      'uncertain', 'unsure', 'hindi sigurado', 'dread', 'warning', 'babala',
      'incoming', 'approaching', 'papalapit', 'threatening', 'threat', 'banta'
    ],
    intensifiers: ['getting', 'becoming', 'more', 'increasing', 'growing', 'lumalakas'],
    contextual: ['might', 'could', 'possibly', 'baka', 'siguro', 'posible'],
    weight: 1.5
  },
  'Disbelief': {
    keywords: [
      'unbelievable', 'impossible', 'hindi kapani-paniwala', 'di makapaniwala',
      'shocked', 'stunned', 'nagulat', 'nagugulat', 'cannot believe', 'di matanggap',
      'how could', 'why would', 'bakit ganun', 'paano nangyari', 'unexpected',
      'hindi inaasahan', 'surprising', 'nakakagulat', 'grabe'
    ],
    intensifiers: ['totally', 'completely', 'absolutely', 'lubos', 'sobrang'],
    contextual: ['never thought', 'first time', 'unprecedented', 'unusual'],
    weight: 1.3
  },
  'Resilience': {
    keywords: [
      'strong', 'brave', 'hope', 'malakas', 'matapang', 'pag-asa', 'kakayanin',
      'survive', 'overcome', 'lalaban', 'fight', 'recover', 'rebuild', 'help',
      'support', 'tulong', 'together', 'sama-sama', 'bayanihan', 'community',
      'volunteers', 'rescue', 'saved', 'safe', 'ligtas', 'evacuated', 'shelter'
    ],
    intensifiers: ['will', 'shall', 'must', 'dapat', 'kailangan', 'always'],
    contextual: ['we can', 'we will', 'kaya natin', 'magtulungan', 'unity'],
    weight: 1.8
  },
  'Neutral': {
    keywords: [
      'information', 'update', 'announcement', 'balita', 'impormasyon', 'advisory',
      'report', 'status', 'situation', 'current', 'kasalukuyan', 'official',
      'notice', 'alert', 'bulletin', 'news', 'reported', 'according'
    ],
    intensifiers: ['please', 'kindly', 'pakiusap', 'paki'],
    contextual: ['as of', 'currently', 'ngayon', 'latest'],
    weight: 1.0
  }
};

// Contextual disaster indicators for better detection
const disasterContexts = {
  Earthquake: {
    primaryIndicators: ['earthquake', 'lindol', 'quake', 'magnitude', 'aftershock', 'tremor'],
    locationIndicators: ['epicenter', 'fault line', 'ground', 'building', 'structure'],
    intensityWords: ['strong', 'powerful', 'massive', 'malakas', 'devastating'],
    weight: 2.0
  },
  Flood: {
    primaryIndicators: ['flood', 'baha', 'tubig', 'water level', 'rising', 'overflow'],
    locationIndicators: ['street', 'road', 'area', 'community', 'river', 'dam'],
    intensityWords: ['deep', 'rising', 'severe', 'malalim', 'lumalalim'],
    weight: 1.8
  },
  Typhoon: {
    primaryIndicators: ['typhoon', 'bagyo', 'storm', 'wind', 'rain', 'signal'],
    locationIndicators: ['eye', 'path', 'track', 'landfall', 'coastal', 'area'],
    intensityWords: ['strong', 'intense', 'super', 'powerful', 'malakas'],
    weight: 1.9
  }
};

// Function to analyze emotion with context
function analyzeEmotionWithContext(text: string, disasterType: string | null | undefined): {
  emotion: string;
  confidence: number;
  explanation: string;
} {
  const textLower = text.toLowerCase();
  let scores: Record<string, number> = {};
  let explanations: Record<string, string[]> = {};

  // Calculate base emotion scores with context
  for (const [emotion, pattern] of Object.entries(emotionPatterns)) {
    let score = 0;
    let reasons: string[] = [];

    // Check keywords
    pattern.keywords.forEach(keyword => {
      const matches = (textLower.match(new RegExp(keyword, 'g')) || []).length;
      if (matches > 0) {
        score += matches * pattern.weight;
        reasons.push(`Found "${keyword}" ${matches} time(s)`);
      }
    });

    // Check intensifiers
    pattern.intensifiers.forEach(intensifier => {
      const matches = (textLower.match(new RegExp(intensifier, 'g')) || []).length;
      if (matches > 0) {
        score += matches * 0.5 * pattern.weight;
        reasons.push(`Intensifier "${intensifier}" present`);
      }
    });

    // Check contextual patterns
    pattern.contextual.forEach(context => {
      const matches = (textLower.match(new RegExp(context, 'g')) || []).length;
      if (matches > 0) {
        score += matches * 0.7 * pattern.weight;
        reasons.push(`Contextual pattern "${context}" found`);
      }
    });

    scores[emotion] = score;
    explanations[emotion] = reasons;
  }

  // Apply disaster context boost
  const normalizedDisasterType = disasterType ?? null;
  if (normalizedDisasterType) {
    const disasterContext = disasterContexts[normalizedDisasterType as keyof typeof disasterContexts];
    if (disasterContext) {
      // Boost emotional scores based on disaster context
      disasterContext.primaryIndicators.forEach(indicator => {
        if (textLower.includes(indicator)) {
          scores['Fear/Anxiety'] *= 1.2;
          scores['Panic'] *= 1.3;
        }
      });

      disasterContext.intensityWords.forEach(word => {
        if (textLower.includes(word)) {
          scores['Fear/Anxiety'] *= 1.1;
          scores['Panic'] *= 1.2;
        }
      });
    }
  }

  // Find dominant emotion
  let maxScore = 0;
  let dominantEmotion = 'Neutral';
  for (const [emotion, score] of Object.entries(scores)) {
    if (score > maxScore) {
      maxScore = score;
      dominantEmotion = emotion;
    }
  }

  // Calculate confidence (normalized score)
  const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);
  const confidence = totalScore > 0 ? (maxScore / totalScore) : 0.5;

  // Generate detailed explanation
  const explanation = `Primary emotion detected: ${dominantEmotion} (Confidence: ${(confidence * 100).toFixed(1)}%)
Reasons: ${explanations[dominantEmotion].join(', ')}
${normalizedDisasterType ? `Disaster Context: ${normalizedDisasterType} - Enhanced emotional analysis applied` : ''}
Contributing factors: ${Object.entries(scores)
    .filter(([emotion, score]) => score > 0 && emotion !== dominantEmotion)
    .map(([emotion, score]) => `${emotion} (${(score / totalScore * 100).toFixed(1)}%)`)
    .join(', ')}`;

  return {
    emotion: dominantEmotion,
    confidence,
    explanation
  };
}


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

  // Update file upload endpoint with enhanced emotion analysis
  app.post('/api/upload-csv', upload.single('file'), async (req: Request, res: Response) => {
    let sessionId: string | undefined;
    let updateProgress: ((processed: number, stage: string, error?: string) => void) | undefined;

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

      // Process the CSV with enhanced emotion analysis
      const { data, storedFilename, recordCount } = await pythonService.processCSV(
        fileBuffer,
        originalFilename,
        (processed: number, stage: string) => {
          updateProgress?.(processed, `Analyzing emotions in raw data: ${stage}`);
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

      // Enhanced processing of sentiment posts with emotion analysis
      const sentimentPosts = await Promise.all(
        data.results.map(async (result: any) => {
          const emotionAnalysis = analyzeEmotionWithContext(result.text, result.disasterType);

          return storage.createSentimentPost(
            insertSentimentPostSchema.parse({
              text: result.text,
              timestamp: new Date(result.timestamp),
              source: `CSV Upload: ${originalFilename}`,
              language: result.language,
              sentiment: emotionAnalysis.emotion,
              confidence: emotionAnalysis.confidence,
              explanation: emotionAnalysis.explanation,
              location: result.location || null,
              disasterType: result.disasterType || null,
              fileId: analyzedFile.id
            })
          );
        })
      );

      // Prioritize disaster event generation for uploaded data
      await generateDisasterEvents(sentimentPosts);

      // Final progress update
      updateProgress?.(totalRecords, 'Analysis complete');

      res.json({
        file: analyzedFile,
        posts: sentimentPosts,
        metrics: {
          ...data.metrics,
          emotionBreakdown: sentimentPosts.reduce((acc: Record<string, number>, post) => {
            acc[post.sentiment] = (acc[post.sentiment] || 0) + 1;
            return acc;
          }, {}),
          averageConfidence: sentimentPosts.reduce((sum, post) => sum + post.confidence, 0) / sentimentPosts.length
        }
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
          if (sessionId) {  // Additional check to satisfy TypeScript
            uploadProgressMap.delete(sessionId);
          }
        }, 5000);
      }
    }
  });

  // Analyze text (single or batch)
  app.post('/api/analyze-text', async (req: Request, res: Response) => {
    try {
      const { text, texts, source = 'Manual Input' } = req.body;

      if (!text && (!texts || !Array.isArray(texts) || texts.length === 0)) {
        return res.status(400).json({ error: "No text provided. Send either 'text' or 'texts' array in the request body" });
      }

      // Process single text
      if (text) {
        const result = await pythonService.analyzeSentiment(text);
        const emotionAnalysis = analyzeEmotionWithContext(text, result.disasterType ?? null);

        const sentimentPost = await storage.createSentimentPost(
          insertSentimentPostSchema.parse({
            text,
            timestamp: new Date(),
            source,
            language: result.language,
            sentiment: emotionAnalysis.emotion,
            confidence: emotionAnalysis.confidence,
            explanation: emotionAnalysis.explanation,
            location: result.location || null,
            disasterType: result.disasterType || null,
            fileId: null
          })
        );

        return res.json({
          post: sentimentPost,
          analysis: emotionAnalysis
        });
      }

      // Process multiple texts with enhanced analysis
      const processResults = await Promise.all(texts.map(async (textItem: string) => {
        const result = await pythonService.analyzeSentiment(textItem);
        const emotionAnalysis = analyzeEmotionWithContext(textItem, result.disasterType ?? null);

        const post = await storage.createSentimentPost(
          insertSentimentPostSchema.parse({
            text: textItem,
            timestamp: new Date(),
            source,
            language: result.language,
            sentiment: emotionAnalysis.emotion,
            confidence: emotionAnalysis.confidence,
            explanation: emotionAnalysis.explanation,
            location: result.location || null,
            disasterType: result.disasterType || null,
            fileId: null
          })
        );

        return {
          post,
          analysis: emotionAnalysis
        };
      }));

      // Process uploaded file data with priority
      if (source.includes('CSV') || source.includes('Upload')) {
        await generateDisasterEvents(processResults.map(r => r.post));
      }

      res.json({
        results: processResults,
        summary: {
          totalAnalyzed: processResults.length,
          emotionBreakdown: processResults.reduce((acc: Record<string, number>, curr) => {
            const emotion = curr.analysis.emotion;
            acc[emotion] = (acc[emotion] || 0) + 1;
            return acc;
          }, {}),
          averageConfidence: processResults.reduce((sum, curr) => sum + curr.analysis.confidence, 0) / processResults.length
        }
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