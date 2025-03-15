import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { nanoid } from 'nanoid';
import { log } from './vite';

interface ProcessCSVResult {
  results: {
    text: string;
    timestamp: string;
    source: string;
    language: string;
    sentiment: string;
    confidence: number;
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
  private maxRetries: number = 3;
  private retryDelay: number = 1000;

  constructor() {
    this.pythonBinary = 'python3';
    this.tempDir = path.join(os.tmpdir(), 'disaster-sentiment');
    this.scriptPath = path.join(process.cwd(), 'server', 'python', 'process.py');

    // Create temp directory if it doesn't exist
    if (!fs.existsSync(this.tempDir)) {
      fs.mkdirSync(this.tempDir, { recursive: true });
    }
  }

  public async processCSV(
    fileBuffer: Buffer, 
    originalFilename: string,
    onProgress?: (processed: number, stage: string) => void
  ): Promise<{
    data: ProcessCSVResult,
    storedFilename: string,
    recordCount: number
  }> {
    const uniqueId = nanoid();
    const storedFilename = `${uniqueId}-${originalFilename}`;
    const tempFilePath = path.join(this.tempDir, storedFilename);

    try {
      // Validate file content before writing
      const content = fileBuffer.toString('utf-8');
      const lines = content.split('\n');

      if (lines.length < 2) {
        throw new Error('CSV file appears to be empty or malformed');
      }

      fs.writeFileSync(tempFilePath, fileBuffer);

      log(`Processing CSV file: ${originalFilename}`, 'python-service');

      let result;
      try {
        result = await this.runPythonScript(tempFilePath, '', onProgress);
      } catch (error) {
        log(`CSV processing failed: ${error}`, 'python-service');
        throw new Error(`Failed to process CSV file: ${error}`);
      }

      if (!result) {
        throw new Error('Failed to process CSV file: No result returned from Python script');
      }

      try {
        const data = JSON.parse(result) as ProcessCSVResult;

        if (!data.results || !Array.isArray(data.results)) {
          throw new Error('Invalid data format returned from Python script');
        }

        // Log details about what was processed
        log(`Successfully processed ${data.results.length} records from CSV`, 'python-service');

        // Count how many records have location from the CSV
        const withLocation = data.results.filter(r => r.location).length;
        const withDisasterType = data.results.filter(r => r.disasterType && r.disasterType !== "Not Specified").length;

        log(`Records with location: ${withLocation}/${data.results.length}`, 'python-service');
        log(`Records with disaster type: ${withDisasterType}/${data.results.length}`, 'python-service');

        return {
          data,
          storedFilename,
          recordCount: data.results.length
        };
      } catch (error) {
        log(`Error parsing Python script output: ${error}`, 'python-service');
        throw new Error('Failed to parse Python script output');
      }
    } catch (error) {
      // Clean up temp file on error
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
      throw error;
    }
  }

  private runPythonScript(
    filePath: string = '', 
    textToAnalyze: string = '',
    onProgress?: (processed: number, stage: string) => void
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      const args = [this.scriptPath];

      if (filePath) {
        args.push('--file', filePath);
      }

      if (textToAnalyze) {
        args.push('--text', textToAnalyze);
      }

      log(`Running Python script with args: ${args.join(' ')}`, 'python-service');

      const pythonProcess = spawn(this.pythonBinary, args);

      let output = '';
      let errorOutput = '';
      let lastProgressUpdate = Date.now();

      pythonProcess.stdout.on('data', (data) => {
        const dataStr = data.toString();

        // Handle progress updates with rate limiting
        if (onProgress && dataStr.includes('PROGRESS:')) {
          const now = Date.now();
          if (now - lastProgressUpdate >= 100) { // Update every 100ms max
            try {
              const progressData = JSON.parse(dataStr.split('PROGRESS:')[1]);
              onProgress(progressData.processed, progressData.stage);
              lastProgressUpdate = now;
            } catch (e) {
              // Log but don't fail on progress parsing errors
              log(`Progress parsing error: ${e}`, 'python-service');
            }
          }
        } else {
          output += dataStr;
        }
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
        log(`Python process error: ${data.toString()}`, 'python-service');
      });

      // Set a timeout for the entire process
      const timeout = setTimeout(() => {
        pythonProcess.kill();
        reject(new Error('Python script execution timed out after 5 minutes'));
      }, 5 * 60 * 1000); // 5 minutes timeout

      pythonProcess.on('close', (code) => {
        clearTimeout(timeout);

        if (code !== 0) {
          log(`Python process error: ${errorOutput}`, 'python-service');
          reject(new Error(`Python script exited with code ${code}: ${errorOutput}`));
          return;
        }

        const trimmedOutput = output.trim();

        if (!trimmedOutput) {
          log(`Error: Python process returned empty output`, 'python-service');
          reject(new Error('Python process returned empty output'));
          return;
        }

        try {
          // Verify output is valid JSON
          JSON.parse(trimmedOutput);
          resolve(trimmedOutput);
        } catch (e) {
          log(`Invalid JSON output: ${trimmedOutput}`, 'python-service');
          log(`JSON parse error: ${e}`, 'python-service');

          reject(new Error('Invalid JSON output from Python script'));
        }
      });

      pythonProcess.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`Failed to start Python process: ${error.message}`));
      });
    });
  }
  public async analyzeSentiment(
    text: string, 
    csvLocation?: string, 
    csvEmotion?: string, 
    csvDisasterType?: string
  ): Promise<{
    sentiment: string; 
    confidence: number;
    explanation: string;
    language: string;
    disasterType?: string;
    location?: string;
  }> {
    // Run the Python script only once - don't retry
    // The Python script itself will manage key rotation to handle rate limits
    
    // Create a JSON object with the parameters
    const params = {
      text,
      csvLocation,
      csvEmotion,
      csvDisasterType
    };
    
    const paramsJson = JSON.stringify(params);
    
    try {
      // We pass the JSON parameters as the text parameter
      const result = await this.runPythonScript('', paramsJson);
      return JSON.parse(result);
    } catch (error) {
      log(`Sentiment analysis failed: ${error}`, 'python-service');
      throw new Error(`Failed to analyze sentiment: ${error}`);
    }
  }
}

export const pythonService = new PythonService();