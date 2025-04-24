/**
 * Hybrid Model Processor Integration
 * This module provides integration between the Node.js application and the Python hybrid model
 */

import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { nanoid } from 'nanoid';

// Track active processing sessions
const activeSessions = new Map();

/**
 * HybridModelProcessor class
 * Handles sending CSV files to the Python hybrid model for processing
 */
export class HybridModelProcessor {
  constructor() {
    this.pythonBinary = process.env.PYTHON_PATH || 'python';
    this.scriptPath = this.findProcessorScript();
    this.processingCounter = 0;
    
    console.log(`HybridModelProcessor initialized with Python binary: ${this.pythonBinary}`);
    console.log(`Processor script path: ${this.scriptPath}`);
    
    // Create models directory if it doesn't exist
    const modelsDir = path.join(process.cwd(), 'models');
    if (!fs.existsSync(modelsDir)) {
      fs.mkdirSync(modelsDir, { recursive: true });
      console.log(`Created models directory at ${modelsDir}`);
    }
  }

  /**
   * Find the processor script path
   * @returns {string} Path to the processor script
   */
  findProcessorScript() {
    // Check for the script in the python directory
    const scriptPath = path.join(process.cwd(), 'python', 'process_csv_hybrid.py');
    
    if (fs.existsSync(scriptPath)) {
      return scriptPath;
    }
    
    // Return a default path if the script doesn't exist yet
    return path.join(process.cwd(), 'python', 'process_csv_hybrid.py');
  }

  /**
   * Process a CSV file using the hybrid model
   * @param {string} filePath - Path to the CSV file
   * @param {string} textColumn - Name of the column containing text data
   * @param {boolean} validate - Whether to validate with ground truth data
   * @returns {Promise<object>} - Processing results and metrics
   */
  async processCSV(filePath, textColumn = 'text', validate = false) {
    return new Promise((resolve, reject) => {
      // Generate a unique session ID if not provided
      const sessionId = nanoid(8);
      
      console.log(`Starting hybrid model processing for file: ${filePath} (Session ID: ${sessionId})`);
      
      // Prepare arguments for the Python script
      const args = [
        this.scriptPath,
        '--input', filePath,
        '--text-column', textColumn
      ];
      
      if (validate) {
        args.push('--validate');
      }
      
      // Include session ID for tracking
      args.push('--session-id', sessionId);
      
      // Store output data
      let jsonOutput = '';
      let errorOutput = '';
      let lastProgress = {};
      
      // Track processing statistics
      const processingInfo = {
        sessionId,
        filePath,
        startTime: Date.now(),
        lastProgressUpdate: Date.now(),
        progress: {
          processed: 0,
          total: 0,
          stage: 'Initializing...'
        },
        processId: this.processingCounter++
      };
      
      // Spawn the Python process
      const pythonProcess = spawn(this.pythonBinary, args);
      
      // Add to active sessions
      activeSessions.set(sessionId, {
        process: pythonProcess,
        info: processingInfo
      });
      
      // Handle process output
      pythonProcess.stdout.on('data', (data) => {
        const output = data.toString();
        
        // Check if this is a progress update
        if (output.includes('PROGRESS:')) {
          try {
            // Extract JSON progress data
            const progressMatch = output.match(/PROGRESS:(.*?)::END_PROGRESS/);
            if (progressMatch && progressMatch[1]) {
              const progressData = JSON.parse(progressMatch[1]);
              
              // Update processing info with progress
              processingInfo.progress = progressData;
              processingInfo.lastProgressUpdate = Date.now();
              
              // Store last progress for result calculation
              lastProgress = progressData;
              
              console.log(`Hybrid model progress update for session ${sessionId}: ${progressData.stage}`);
            }
          } catch (error) {
            console.error(`Error parsing progress data: ${error.message}`);
          }
        } else if (output.includes('RESULT:')) {
          // Extract final JSON result
          try {
            const resultMatch = output.match(/RESULT:(.*?)::END_RESULT/);
            if (resultMatch && resultMatch[1]) {
              jsonOutput = resultMatch[1];
            }
          } catch (error) {
            console.error(`Error parsing result data: ${error.message}`);
          }
        } else {
          // Regular console output
          console.log(`[Hybrid Model ${sessionId}] ${output.trim()}`);
        }
      });
      
      // Handle errors
      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
        console.error(`[Hybrid Model ${sessionId} Error] ${data.toString().trim()}`);
      });
      
      // Handle process completion
      pythonProcess.on('close', (code) => {
        // Remove from active sessions
        activeSessions.delete(sessionId);
        
        console.log(`Hybrid model processing complete for session ${sessionId} with exit code: ${code}`);
        
        if (code === 0 && jsonOutput) {
          try {
            // Parse the JSON output
            const result = JSON.parse(jsonOutput);
            
            // Calculate processing time
            const processingTime = (Date.now() - processingInfo.startTime) / 1000; // seconds
            
            // Add session details to result
            result.sessionId = sessionId;
            result.processingTime = processingTime;
            result.totalProcessed = lastProgress.processed || result.results?.length || 0;
            
            resolve(result);
          } catch (error) {
            reject(new Error(`Failed to parse JSON output: ${error.message}`));
          }
        } else {
          reject(new Error(`Hybrid model processing failed with code ${code}: ${errorOutput}`));
        }
      });
      
      // Handle unexpected errors
      pythonProcess.on('error', (error) => {
        activeSessions.delete(sessionId);
        reject(new Error(`Failed to start hybrid model processing: ${error.message}`));
      });
    });
  }

  /**
   * Cancel processing for a session
   * @param {string} sessionId - Session ID to cancel
   * @returns {boolean} - Whether cancellation was successful
   */
  cancelProcessing(sessionId) {
    if (activeSessions.has(sessionId)) {
      const session = activeSessions.get(sessionId);
      
      // Kill the process
      if (session.process) {
        session.process.kill('SIGTERM');
        console.log(`Cancelled hybrid model processing for session ${sessionId}`);
      }
      
      // Remove from active sessions
      activeSessions.delete(sessionId);
      return true;
    }
    
    return false;
  }

  /**
   * Get active processing sessions
   * @returns {Array} List of active session IDs
   */
  getActiveSessions() {
    return Array.from(activeSessions.keys());
  }

  /**
   * Get detailed information about active processing sessions
   * @returns {Array} List of session information objects
   */
  getActiveSessionsInfo() {
    return Array.from(activeSessions.entries()).map(([sessionId, session]) => {
      const { info } = session;
      const elapsedTime = (Date.now() - info.startTime) / 1000; // seconds
      
      return {
        sessionId,
        filePath: info.filePath,
        startTime: new Date(info.startTime).toISOString(),
        elapsedTime,
        progress: info.progress,
        processId: info.processId
      };
    });
  }

  /**
   * Get progress for a specific session
   * @param {string} sessionId - Session ID to get progress for
   * @returns {object|null} Progress information or null if session not found
   */
  getSessionProgress(sessionId) {
    if (activeSessions.has(sessionId)) {
      return activeSessions.get(sessionId).info.progress;
    }
    
    return null;
  }

  /**
   * Cancel all active processing sessions
   * @returns {number} Number of sessions canceled
   */
  cancelAllSessions() {
    let cancelCount = 0;
    
    for (const sessionId of activeSessions.keys()) {
      if (this.cancelProcessing(sessionId)) {
        cancelCount++;
      }
    }
    
    return cancelCount;
  }
}

export const hybridProcessor = new HybridModelProcessor();