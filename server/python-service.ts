import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { nanoid } from 'nanoid';
import { log } from './vite';
import { usageTracker } from './utils/usage-tracker';
import { storage } from './storage';

// Global array to store console logs from Python service
export const pythonConsoleMessages: {message: string, timestamp: Date}[] = [];

interface ProcessCSVResult {
  results: {
    text: string;
    timestamp: string;
    source: string;
    language: string;
    sentiment: string;
    confidence: number;  // This will now be the real AI confidence score
    explanation?: string;
    disasterType?: string;
    location?: string;
  }[];
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  };
}

export class PythonService {
  private pythonBinary: string;
  private tempDir: string;
  private scriptPath: string;
  private confidenceCache: Map<string, number>;  // Cache for confidence scores
  private similarityCache: Map<string, boolean>; // Cache for text similarity checks

  constructor() {
    this.pythonBinary = 'python3';
    this.tempDir = path.join(os.tmpdir(), 'disaster-sentiment');
    this.scriptPath = path.join(process.cwd(), 'server', 'python', 'process.py');
    this.confidenceCache = new Map();  // Initialize confidence cache
    this.similarityCache = new Map();  // Initialize similarity cache

    if (!fs.existsSync(this.tempDir)) {
      fs.mkdirSync(this.tempDir, { recursive: true });
    }
  }
  
  // Utility method to extract disaster type from text
  private extractDisasterTypeFromText(text: string): string | null {
    const textLower = text.toLowerCase();
    
    // Check for typhoon/bagyo
    if (textLower.includes('typhoon') || textLower.includes('bagyo') || 
        textLower.includes('cyclone') || textLower.includes('storm')) {
      return 'Typhoon';
    }
    
    // Check for flood/baha
    if (textLower.includes('flood') || textLower.includes('baha') || 
        textLower.includes('tubig') || textLower.includes('water rising')) {
      return 'Flood';
    }
    
    // Check for earthquake/lindol
    if (textLower.includes('earthquake') || textLower.includes('lindol') || 
        textLower.includes('linog') || textLower.includes('magnitude') || 
        textLower.includes('shaking') || textLower.includes('tremor')) {
      return 'Earthquake';
    }
    
    // Check for volcano/bulkan
    if (textLower.includes('volcano') || textLower.includes('bulkan') || 
        textLower.includes('eruption') || textLower.includes('lava') || 
        textLower.includes('ash fall') || textLower.includes('magma')) {
      return 'Volcanic Eruption';
    }
    
    // Check for fire/sunog
    if (textLower.includes('fire') || textLower.includes('sunog') || 
        textLower.includes('burning') || textLower.includes('nasusunog')) {
      return 'Fire';
    }
    
    // Check for landslide/pagguho
    if (textLower.includes('landslide') || textLower.includes('pagguho') || 
        textLower.includes('mudslide') || textLower.includes('rockfall') || 
        textLower.includes('gumuho')) {
      return 'Landslide';
    }
    
    return null;
  }
  
  // Utility method to extract location from text
  private extractLocationFromText(text: string): string | null {
    // Philippine locations - major cities and regions
    const locations = [
      'Manila', 'Quezon City', 'Davao', 'Cebu', 'Makati', 'Taguig', 'Pasig',
      'Cagayan', 'Bicol', 'Samar', 'Leyte', 'Tacloban', 'Batanes', 'Mindanao',
      'Luzon', 'Visayas', 'Palawan', 'Mindoro', 'Batangas', 'Cavite', 'Laguna',
      'Albay', 'Baguio', 'Zambales', 'Pampanga', 'Bulacan', 'Iloilo', 'Bacolod',
      'Zamboanga', 'General Santos', 'Cagayan de Oro', 'Butuan', 'Camarines'
    ];
    
    // Convert to lowercase for case-insensitive matching
    const textLower = text.toLowerCase();
    
    // Check each location
    for (const location of locations) {
      if (textLower.includes(location.toLowerCase())) {
        return location;
      }
    }
    
    // Check for "sa" + location pattern in Filipino
    const saPattern = /\bsa\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b/;
    const saMatch = text.match(saPattern);
    if (saMatch && saMatch.length > 1) {
      return saMatch[1]; // Return the first captured group
    }
    
    return null;
  }

  private getCachedConfidence(text: string): number | undefined {
    return this.confidenceCache.get(text);
  }

  private setCachedConfidence(text: string, confidence: number): void {
    this.confidenceCache.set(text, confidence);
  }
  
  // Method to clear cache for a specific text
  public clearCacheForText(text: string): void {
    if (this.confidenceCache.has(text)) {
      log(`Clearing cache entry for text: "${text.substring(0, 30)}..."`, 'python-service');
      this.confidenceCache.delete(text);
    }
  }
  
  // Method to clear the entire cache
  public clearCache(): void {
    log(`Clearing entire confidence cache (${this.confidenceCache.size} entries)`, 'python-service');
    this.confidenceCache.clear();
    this.similarityCache.clear();
  }
  
  // New method to analyze if two texts have similar semantic meaning
  public async analyzeSimilarityForFeedback(
    text1: string, 
    text2: string
  ): Promise<{
    areSimilar: boolean;
    score: number;
    explanation?: string;
  }> {
    try {
      // Create a cache key for these two texts
      const cacheKey = `${text1.trim().toLowerCase()}|${text2.trim().toLowerCase()}`;
      
      // Check cache first
      if (this.similarityCache.has(cacheKey)) {
        const cached = this.similarityCache.get(cacheKey);
        return {
          areSimilar: cached === true,
          score: cached === true ? 0.95 : 0.2,
          explanation: cached === true ? 
            "Cached result: Texts have similar semantic meaning" : 
            "Cached result: Texts have different semantic meanings"
        };
      }
      
      // Simple rule-based check for exact match
      if (text1.trim().toLowerCase() === text2.trim().toLowerCase()) {
        this.similarityCache.set(cacheKey, true);
        return {
          areSimilar: true,
          score: 1.0,
          explanation: "Exact match"
        };
      }
      
      // Check if one text contains the other (ignoring case)
      if (text1.trim().toLowerCase().includes(text2.trim().toLowerCase()) || 
          text2.trim().toLowerCase().includes(text1.trim().toLowerCase())) {
        this.similarityCache.set(cacheKey, true);
        return {
          areSimilar: true,
          score: 0.9,
          explanation: "One text contains the other"
        };
      }
      
      // If one has "joke" or "eme" and the other doesn't, they're likely different
      const jokeWords = ['joke', 'eme', 'charot', 'just kidding', 'kidding', 'lol', 'haha'];
      const text1HasJoke = jokeWords.some(word => text1.toLowerCase().includes(word));
      const text2HasJoke = jokeWords.some(word => text2.toLowerCase().includes(word));
      
      if (text1HasJoke !== text2HasJoke) {
        // One has joke indicators and the other doesn't - likely different meanings
        this.similarityCache.set(cacheKey, false);
        return {
          areSimilar: false,
          score: 0.1,
          explanation: "One text contains joke indicators while the other doesn't"
        };
      }
      
      // Check if both contain negation words
      const negationWords = ['hindi', 'wala', 'walang', 'not', "isn't", "aren't", "wasn't", "didn't", "doesn't", "won't"];
      const text1HasNegation = negationWords.some(word => text1.toLowerCase().includes(word));
      const text2HasNegation = negationWords.some(word => text2.toLowerCase().includes(word));
      
      if (text1HasNegation !== text2HasNegation) {
        // One has negation and the other doesn't - likely different meanings
        this.similarityCache.set(cacheKey, false);
        return {
          areSimilar: false,
          score: 0.15,
          explanation: "One text contains negation while the other doesn't"
        };
      }
      
      // Calculate word overlap 
      const words1 = new Set(text1.toLowerCase().match(/\b\w+\b/g) || []);
      const words2 = new Set(text2.toLowerCase().match(/\b\w+\b/g) || []);
      
      // Find common words
      const commonWords = [...words1].filter(word => words2.has(word));
      
      // If there are enough common significant words, they might be similar
      if (commonWords.length >= 4 && 
          (commonWords.length / Math.max(words1.size, words2.size)) > 0.6) {
        this.similarityCache.set(cacheKey, true);
        return {
          areSimilar: true,
          score: 0.8,
          explanation: "Texts share significant common words and context"
        };
      }
      
      // Default to not similar for now
      this.similarityCache.set(cacheKey, false);
      return {
        areSimilar: false,
        score: 0.3,
        explanation: "Texts don't share enough common context to determine similarity"
      };
    } catch (error) {
      log(`Error analyzing text similarity: ${error}`, 'python-service');
      return {
        areSimilar: false,
        score: 0,
        explanation: `Error analyzing similarity: ${error}`
      };
    }
  }

  // Process feedback and train the model with corrections
  public async trainModelWithFeedback(
    originalText: string, 
    originalSentiment: string, 
    correctedSentiment: string
  ): Promise<{
    status: string;
    message: string;
    performance?: {
      previous_accuracy: number;
      new_accuracy: number;
      improvement: number;
    }
  }> {
    try {
      log(`Training model with feedback: "${originalText.substring(0, 30)}..." - ${originalSentiment} → ${correctedSentiment}`, 'python-service');
      
      // Create feedback payload - ensure it's properly formatted with double quotes for JSON parsing on Python side
      const feedbackData = JSON.stringify({
        feedback: true,
        originalText,
        originalSentiment,
        correctedSentiment
      });
      
      // Log the exact payload being sent to Python for debugging
      log(`Sending feedback payload to Python: ${feedbackData}`, 'python-service');
      
      return new Promise((resolve, reject) => {
        // Use correct argument format - use proper string for --text parameter
        const pythonProcess = spawn(this.pythonBinary, [
          this.scriptPath,
          '--text', feedbackData
        ]);
        
        let outputData = '';
        let errorData = '';
        
        pythonProcess.stdout.on('data', (data) => {
          const dataString = data.toString();
          outputData += dataString;
          log(`Python stdout: ${dataString.trim()}`, 'python-service');
          
          // Save to console messages for debugging
          pythonConsoleMessages.push({
            message: dataString.trim(),
            timestamp: new Date()
          });
        });
        
        pythonProcess.stderr.on('data', (data) => {
          const message = data.toString().trim();
          errorData += message;
          
          // Log all Python process output for debugging
          pythonConsoleMessages.push({
            message: message,
            timestamp: new Date()
          });
          
          log(`Python process error: ${message}`, 'python-service');
        });
        
        pythonProcess.on('close', (code) => {
          if (code !== 0) {
            log(`Python process exited with code ${code}`, 'python-service');
            return reject(new Error(`Python process failed with code ${code}: ${errorData}`));
          }
          
          // Trim the output to remove any whitespace
          const trimmedOutput = outputData.trim();
          log(`Raw Python output: "${trimmedOutput}"`, 'python-service');
          
          try {
            // Ensure we're parsing valid JSON
            const result = JSON.parse(trimmedOutput);
            log(`Model training result: ${result.status} - ${result.message}`, 'python-service');
            
            if (result.status === 'success') {
              // Purge from cache if we've updated the model
              this.confidenceCache.delete(originalText);
            }
            
            // Return a successful result
            resolve(result);
          } catch (err) {
            const parseError = `Failed to parse Python output: ${err}. Raw output: "${trimmedOutput}"`;
            log(parseError, 'python-service');
            reject(new Error(parseError));
          }
        });
        
        // Handle process error events
        pythonProcess.on('error', (err) => {
          const errorMsg = `Error spawning Python process: ${err}`;
          log(errorMsg, 'python-service');
          reject(new Error(errorMsg));
        });
      });
    } catch (error) {
      const errorMsg = `Error training model: ${error}`;
      log(errorMsg, 'python-service');
      
      // Return a structured error response
      return {
        status: "error",
        message: errorMsg
      };
    }
  }

  public async processCSV(
    fileBuffer: Buffer, 
    originalFilename: string,
    onProgress?: (processed: number, stage: string, total?: number) => void
  ): Promise<{
    data: ProcessCSVResult,
    storedFilename: string,
    recordCount: number
  }> {
    const uniqueId = nanoid();
    const storedFilename = `${uniqueId}-${originalFilename}`;
    const tempFilePath = path.join(this.tempDir, storedFilename);

    try {
      const content = fileBuffer.toString('utf-8');
      const lines = content.split('\n');
      const totalRecords = lines.length - 1;

      if (lines.length < 2) {
        throw new Error('CSV file appears to be empty or malformed');
      }
      
      // Check if we can process this file based on the daily limit
      if (usageTracker.hasReachedDailyLimit()) {
        throw new Error('Daily processing limit of 10,000 rows has been reached. Please try again tomorrow.');
      }
      
      // Calculate how many rows we can process
      const processableRowCount = usageTracker.getProcessableRowCount(totalRecords);
      if (processableRowCount === 0) {
        throw new Error('Cannot process any more rows today. Daily limit reached.');
      }
      
      // If we can't process all rows, create a truncated version of the file
      if (processableRowCount < totalRecords) {
        log(`Daily limit restriction: Can only process ${processableRowCount} of ${totalRecords} rows.`, 'python-service');
        // Include header row (line 0) plus processableRowCount number of data rows
        const truncatedContent = lines.slice(0, processableRowCount + 1).join('\n');
        fs.writeFileSync(tempFilePath, truncatedContent);
        
        if (onProgress) {
          onProgress(0, `PROGRESS:{"processed":0,"stage":"Daily limit restriction: Can only process ${processableRowCount} of ${totalRecords} rows.","total":processableRowCount}`, processableRowCount);
        }
      } else {
        fs.writeFileSync(tempFilePath, fileBuffer);
      }
      
      log(`Processing CSV file: ${originalFilename}`, 'python-service');

      const pythonProcess = spawn(this.pythonBinary, [
        this.scriptPath,
        '--file', tempFilePath
      ]);

      const result = await new Promise<string>((resolve, reject) => {
        let output = '';
        let errorOutput = '';

        // Handle progress events from Python script
        pythonProcess.stdout.on('data', (data) => {
          const dataStr = data.toString();
          
          // Store stdout message in our global array
          if (dataStr.trim()) {
            pythonConsoleMessages.push({
              message: dataStr.trim(),
              timestamp: new Date()
            });
            
            // Log to server console for debugging
            log(`Python stdout: ${dataStr.trim()}`, 'python-service');
          }
          
          if (onProgress && dataStr.includes('PROGRESS:')) {
            try {
              const progressData = JSON.parse(dataStr.split('PROGRESS:')[1]);
              const rawMessage = data.toString().trim();
              log(`Progress update: ${JSON.stringify(progressData)}`, 'python-service');
              onProgress(
                progressData.processed,
                rawMessage, // Send the raw message instead of just the stage
                progressData.total
              );
            } catch (e) {
              log(`Progress parsing error: ${e}`, 'python-service');
            }
          } else {
            output += dataStr;
          }
        });

        pythonProcess.stderr.on('data', (data) => {
          const errorMsg = data.toString();
          errorOutput += errorMsg;
          
          // Save all Python console output
          pythonConsoleMessages.push({
            message: errorMsg.trim(),
            timestamp: new Date()
          });
          
          // Also treat error messages as progress updates to show in the UI
          if (onProgress && errorMsg.includes('Completed record')) {
            const matches = errorMsg.match(/Completed record (\d+)\/(\d+)/);
            if (matches) {
              const processed = parseInt(matches[1]);
              const total = parseInt(matches[2]);
              onProgress(processed, errorMsg.trim(), total);
            }
          }
          
          log(`Python process error: ${errorMsg}`, 'python-service');
        });

        // No timeout as requested by user - Python process will run until completion
        

        pythonProcess.on('close', (code) => {
          if (code !== 0) {
            reject(new Error(`Python script exited with code ${code}: ${errorOutput}`));
            return;
          }
          try {
            JSON.parse(output.trim());
            resolve(output.trim());
          } catch (e) {
            reject(new Error('Invalid JSON output from Python script'));
          }
        });

        pythonProcess.on('error', (error) => {
          reject(new Error(`Failed to start Python process: ${error.message}`));
        });
      });

      const data = JSON.parse(result) as ProcessCSVResult;
      if (!data.results || !Array.isArray(data.results)) {
        throw new Error('Invalid data format returned from Python script');
      }

      // Update the usage tracker with the number of rows processed
      usageTracker.incrementRowCount(data.results.length);
      
      log(`Successfully processed ${data.results.length} records from CSV`, 'python-service');
      log(`Daily usage: ${usageTracker.getUsageStats().used}/${usageTracker.getUsageStats().limit} rows`, 'python-service');

      return {
        data,
        storedFilename,
        recordCount: data.results.length
      };

    } catch (error) {
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
      throw error;
    }
  }

  public async analyzeSentiment(text: string): Promise<{
    sentiment: string;
    confidence: number;
    explanation: string;
    language: string;
    disasterType?: string;
    location?: string;
  }> {
    try {
      // First check if we have a saved training example in the database
      // This ensures persistence of training across restarts
      try {
        // Create normalized form of the text (lowercase, space-separated words)
        const textWords = text.toLowerCase().match(/\b\w+\b/g) || [];
        const textKey = textWords.join(' ');
        
        // Try to get a training example from the database
        // First try exact match
        let trainingExample = await storage.getTrainingExampleByText(text);
        
        // If no exact match, try to find partial match based on the core content
        if (!trainingExample) {
          // Clean the text and create a word key
          const textWords = text.toLowerCase().match(/\b\w+\b/g) || [];
          const textKey = textWords.join(' ');
          
          // Get all training examples and check if any are contained in this text
          const allExamples = await storage.getTrainingExamples();
          
          // Try to find a match where the key words from a training example are present in this text
          for (const example of allExamples) {
            const exampleWords = example.text.toLowerCase().match(/\b\w+\b/g) || [];
            const exampleKey = exampleWords.join(' ');
            
            // If the current text contains all the significant words from a training example
            if (exampleWords.length > 3 && textKey.includes(exampleKey)) {
              log(`Found partial match with training example: ${example.sentiment}`, 'python-service');
              trainingExample = example;
              break;
            }
          }
        }
        
        if (trainingExample) {
          log(`Using training example from database for sentiment: ${trainingExample.sentiment}`, 'python-service');
          
          // Custom realistic explanations based on the sentiment
          let explanation = '';
          const disasterType = this.extractDisasterTypeFromText(text) || "UNKNOWN";
          
          // Generate a more realistic AI explanation based on sentiment
          switch(trainingExample.sentiment) {
            case 'Panic':
              explanation = 'The message contains urgent calls for help and extreme concern. The tone indicates panic and immediate distress about the disaster situation.';
              break;
            case 'Fear/Anxiety':
              explanation = 'The message expresses concern and worry about the situation. The language shows anxiety and apprehension about potential impacts.';
              break;
            case 'Disbelief':
              explanation = 'The message expresses shock, surprise or skepticism. The tone indicates the speaker finds the situation unbelievable or is questioning its validity.';
              break;
            case 'Resilience':
              explanation = 'The message shows strength and determination in the face of disaster. The language demonstrates community support and cooperation.';
              break;
            case 'Neutral':
              explanation = 'The message presents information without strong emotional indicators. The tone is informative rather than emotionally charged.';
              break;
            default:
              explanation = 'Analysis indicates significant emotional content related to the disaster situation.';
          }
          
          // Add context about laughter, caps, etc. if present
          if (text.includes('HAHA') || text.includes('haha')) {
            explanation += ' The use of laughter suggests disbelief or nervous humor about the situation.';
          }
          
          if (text.toUpperCase() === text && text.length > 10) {
            explanation += ' The use of all caps indicates heightened emotional intensity.';
          }
          
          // Add context about disaster type if present
          if (disasterType && disasterType !== "UNKNOWN") {
            explanation += ` Context relates to ${disasterType.toLowerCase()} incident.`;
          }
          
          // Translate explanation for Filipino content
          if (trainingExample.language === "Filipino") {
            if (trainingExample.sentiment === 'Panic') {
              explanation = 'Ang mensahe ay naglalaman ng agarang mga panawagan para sa tulong at matinding pag-aalala. Ang tono ay nagpapahiwatig ng pangamba at agarang pangangailangan tungkol sa sitwasyon ng sakuna.';
            } else if (trainingExample.sentiment === 'Fear/Anxiety') {
              explanation = 'Ang mensahe ay nagpapahayag ng pag-aalala tungkol sa sitwasyon. Ang wika ay nagpapakita ng pagkabalisa at pag-aalala tungkol sa mga posibleng epekto.';
            } else if (trainingExample.sentiment === 'Disbelief') {
              explanation = 'Ang mensahe ay nagpapahayag ng gulat, pagkamangha o pagdududa. Ang tono ay nagpapahiwatig na ang nagsasalita ay hindi makapaniwala sa sitwasyon o pinagdududahan ang katotohanan nito.';
            } else if (trainingExample.sentiment === 'Resilience') {
              explanation = 'Ang mensahe ay nagpapakita ng lakas at determinasyon sa harap ng sakuna. Ang wika ay nagpapakita ng suporta at kooperasyon ng komunidad.';
            } else if (trainingExample.sentiment === 'Neutral') {
              explanation = 'Ang mensahe ay nagbibigay ng impormasyon nang walang malakas na mga palatandaan ng emosyon. Ang tono ay nagbibigay-kaalaman sa halip na emosyonal.';
            }
          }
          
          // Return the saved training example results with improved explanation
          return {
            sentiment: trainingExample.sentiment,
            confidence: trainingExample.confidence,
            explanation: explanation,
            language: trainingExample.language,
            disasterType: this.extractDisasterTypeFromText(text) || "UNKNOWN",
            location: this.extractLocationFromText(text) || "UNKNOWN"
          };
        }
      } catch (dbError) {
        // If database lookup fails, log and continue with normal analysis
        log(`Error checking training examples: ${dbError}. Proceeding with API analysis.`, 'python-service');
      }

      // Pass text directly to Python script
      const pythonProcess = spawn(this.pythonBinary, [
        this.scriptPath,
        '--text', text
      ]);

      const result = await new Promise<string>((resolve, reject) => {
        let output = '';
        let errorOutput = '';

        pythonProcess.stdout.on('data', (data) => {
          const dataStr = data.toString();
          output += dataStr;

          if (dataStr.trim()) {
            pythonConsoleMessages.push({
              message: dataStr.trim(),
              timestamp: new Date()
            });

            log(`Python stdout: ${dataStr.trim()}`, 'python-service');
          }
        });

        pythonProcess.stderr.on('data', (data) => {
          const errorMsg = data.toString();
          errorOutput += errorMsg;

          pythonConsoleMessages.push({
            message: errorMsg.trim(),
            timestamp: new Date()
          });

          log(`Python process error: ${errorMsg}`, 'python-service');
        });

        pythonProcess.on('close', (code) => {
          if (code !== 0) {
            reject(new Error(`Python script exited with code ${code}: ${errorOutput}`));
            return;
          }
          resolve(output.trim());
        });
      });

      // Increment usage by 1 for each individual text analysis
      usageTracker.incrementRowCount(1);
      log(`Daily usage: ${usageTracker.getUsageStats().used}/${usageTracker.getUsageStats().limit} rows`, 'python-service');

      const analysisResult = JSON.parse(result);

      // Store the real confidence score in cache
      if (analysisResult.confidence) {
        this.setCachedConfidence(text, analysisResult.confidence);
      }

      // Make sure None values are converted to "UNKNOWN"
      if (!analysisResult.disasterType || analysisResult.disasterType === "None") {
        analysisResult.disasterType = "UNKNOWN";
      }
      
      if (!analysisResult.location || analysisResult.location === "None") {
        analysisResult.location = "UNKNOWN";
      }
      
      return analysisResult;
    } catch (error) {
      log(`Sentiment analysis failed: ${error}`, 'python-service');
      throw new Error(`Failed to analyze sentiment: ${error}`);
    }
  }
}

export const pythonService = new PythonService();