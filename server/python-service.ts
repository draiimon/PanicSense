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

interface BatchProgress {
  batchNumber: number;
  totalBatches: number;
  batchProgress: number;
  stats: {
    successCount: number;
    errorCount: number;
    lastBatchDuration: number;
    averageSpeed: number;
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

    if (!fs.existsSync(this.tempDir)) {
      fs.mkdirSync(this.tempDir, { recursive: true });
    }
  }

  private async retryOperation<T>(
    operation: () => Promise<T>,
    retryCount: number = 0
  ): Promise<T> {
    try {
      return await operation();
    } catch (error) {
      if (retryCount < this.maxRetries) {
        await new Promise(resolve => setTimeout(resolve, this.retryDelay * Math.pow(2, retryCount)));
        return this.retryOperation(operation, retryCount + 1);
      }
      throw error;
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

        pythonProcess.stdout.on('data', (data) => {
          const dataStr = data.toString();
          if (onProgress && dataStr.includes('PROGRESS:')) {
            try {
              const progressData = JSON.parse(dataStr.split('PROGRESS:')[1]);
              onProgress(
                progressData.processed,
                progressData.stage,
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
          errorOutput += data.toString();
          log(`Python process error: ${data.toString()}`, 'python-service');
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

      // Log processing results
      log(`Successfully processed ${data.results.length} records from CSV`, 'python-service');
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
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
      throw error;
    }
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
    const params = {
      text,
      csvLocation,
      csvEmotion,
      csvDisasterType
    };

    const paramsJson = JSON.stringify(params);

    try {
      const result = await this.retryOperation(async () => {
        return this.runPythonScript('', paramsJson);
      });
      return JSON.parse(result);
    } catch (error) {
      log(`Sentiment analysis failed: ${error}`, 'python-service');
      throw new Error(`Failed to analyze sentiment: ${error}`);
    }
  }

  private runPythonScript(
    filePath: string = '', 
    textToAnalyze: string = '',
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

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
        log(`Python process error: ${data.toString()}`, 'python-service');
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

        const trimmedOutput = output.trim();
        if (!trimmedOutput) {
          reject(new Error('Python process returned empty output'));
          return;
        }

        try {
          JSON.parse(trimmedOutput);
          resolve(trimmedOutput);
        } catch (e) {
          reject(new Error('Invalid JSON output from Python script'));
        }
      });

      pythonProcess.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`Failed to start Python process: ${error.message}`));
      });
    });
  }
}

export const pythonService = new PythonService();