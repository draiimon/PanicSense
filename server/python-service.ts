import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { nanoid } from 'nanoid';
import { log } from './vite';
import { usageTracker } from './utils/usage-tracker';

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

  constructor() {
    this.pythonBinary = 'python3';
    this.tempDir = path.join(os.tmpdir(), 'disaster-sentiment');
    this.scriptPath = path.join(process.cwd(), 'server', 'python', 'process.py');
    this.confidenceCache = new Map();  // Initialize confidence cache

    if (!fs.existsSync(this.tempDir)) {
      fs.mkdirSync(this.tempDir, { recursive: true });
    }
  }

  private getCachedConfidence(text: string): number | undefined {
    return this.confidenceCache.get(text);
  }

  private setCachedConfidence(text: string, confidence: number): void {
    this.confidenceCache.set(text, confidence);
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
      // Check cache first
      const cachedConfidence = this.getCachedConfidence(text);
      if (cachedConfidence !== undefined) {
        log(`Using cached confidence score: ${cachedConfidence}`, 'python-service');
        return {
          sentiment: "", // Placeholder -  Real values should come from the cache.  This needs to be populated based on your actual cache structure
          confidence: cachedConfidence,
          explanation: "", // Placeholder
          language: "", // Placeholder
          disasterType: undefined,
          location: undefined
        };
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

      return analysisResult;
    } catch (error) {
      log(`Sentiment analysis failed: ${error}`, 'python-service');
      throw new Error(`Failed to analyze sentiment: ${error}`);
    }
  }
}

export const pythonService = new PythonService();