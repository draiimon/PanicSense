import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { nanoid } from 'nanoid';
import { log } from './vite';

// Global array to store console logs from Python service
export const pythonConsoleMessages: {message: string, timestamp: Date}[] = [];

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

  constructor() {
    this.pythonBinary = 'python3';
    this.tempDir = path.join(os.tmpdir(), 'disaster-sentiment');
    this.scriptPath = path.join(process.cwd(), 'server', 'python', 'process.py');

    if (!fs.existsSync(this.tempDir)) {
      fs.mkdirSync(this.tempDir, { recursive: true });
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

      fs.writeFileSync(tempFilePath, fileBuffer);
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

        const timeout = setTimeout(() => {
          pythonProcess.kill();
          reject(new Error('Python script execution timed out after 5 minutes'));
        }, 5 * 60 * 1000);

        pythonProcess.on('close', (code) => {
          clearTimeout(timeout);
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
          clearTimeout(timeout);
          reject(new Error(`Failed to start Python process: ${error.message}`));
        });
      });

      const data = JSON.parse(result) as ProcessCSVResult;
      if (!data.results || !Array.isArray(data.results)) {
        throw new Error('Invalid data format returned from Python script');
      }

      log(`Successfully processed ${data.results.length} records from CSV`, 'python-service');

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
      // Pass text directly to Python script
      const pythonProcess = spawn(this.pythonBinary, [
        this.scriptPath,
        '--text', text
      ]);

      const result = await new Promise<string>((resolve, reject) => {
        let output = '';
        let errorOutput = '';

        pythonProcess.stdout.on('data', (data) => {
          output += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
          errorOutput += data.toString();
          log(`Python process error: ${data.toString()}`, 'python-service');
        });

        pythonProcess.on('close', (code) => {
          if (code !== 0) {
            reject(new Error(`Python script exited with code ${code}: ${errorOutput}`));
            return;
          }
          resolve(output.trim());
        });
      });

      return JSON.parse(result);
    } catch (error) {
      log(`Sentiment analysis failed: ${error}`, 'python-service');
      throw new Error(`Failed to analyze sentiment: ${error}`);
    }
  }
}

export const pythonService = new PythonService();