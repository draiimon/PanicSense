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
  }[];
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    confusionMatrix: number[][];
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

    // Create temp directory if it doesn't exist
    if (!fs.existsSync(this.tempDir)) {
      fs.mkdirSync(this.tempDir, { recursive: true });
    }
  }

  public async processCSV(fileBuffer: Buffer, originalFilename: string): Promise<{
    data: ProcessCSVResult,
    storedFilename: string,
    recordCount: number
  }> {
    // Save the file to a temporary location
    const uniqueId = nanoid();
    const storedFilename = `${uniqueId}-${originalFilename}`;
    const tempFilePath = path.join(this.tempDir, storedFilename);
    
    fs.writeFileSync(tempFilePath, fileBuffer);
    
    try {
      // Run the Python script with the file
      const result = await this.runPythonScript(tempFilePath);
      
      const data = JSON.parse(result) as ProcessCSVResult;
      
      return {
        data,
        storedFilename,
        recordCount: data.results.length
      };
    } catch (error) {
      // Clean up temp file on error
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
      throw error;
    }
  }

  public async analyzeSentiment(text: string): Promise<{
    sentiment: string; 
    confidence: number;
  }> {
    const result = await this.runPythonScript('', text);
    return JSON.parse(result);
  }

  private runPythonScript(filePath: string = '', textToAnalyze: string = ''): Promise<string> {
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
      });
      
      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          log(`Python process error: ${errorOutput}`, 'python-service');
          reject(new Error(`Python script exited with code ${code}: ${errorOutput}`));
          return;
        }
        
        resolve(output);
      });
    });
  }
}

export const pythonService = new PythonService();
