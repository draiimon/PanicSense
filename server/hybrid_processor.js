/**
 * Hybrid Model Processor Integration
 * This module provides integration between the Node.js application and the Python hybrid model
 */

import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { nanoid } from 'nanoid';
import { pythonService } from './python-service.js';

/**
 * HybridModelProcessor class
 * Handles sending CSV files to the Python hybrid model for processing
 */
export class HybridModelProcessor {
  constructor() {
    this.activeProcesses = new Map(); // Track active processing jobs
    this.pythonBinary = process.env.PYTHON_BINARY || 'python'; // Default Python binary
    
    // Find the correct script path
    this.scriptPath = this.findProcessorScript();
    
    console.log(`Hybrid model processor initialized with script path: ${this.scriptPath}`);
  }
  
  /**
   * Find the processor script path
   * @returns {string} Path to the processor script
   */
  findProcessorScript() {
    const possiblePaths = [
      path.join(process.cwd(), 'python', 'process_csv_hybrid.py'),
      path.join(process.cwd(), 'server', 'python', 'process_csv_hybrid.py'),
    ];
    
    for (const scriptPath of possiblePaths) {
      if (fs.existsSync(scriptPath)) {
        console.log(`✅ Found hybrid processor script at: ${scriptPath}`);
        return scriptPath;
      }
    }
    
    console.error('❌ Could not find hybrid processor script in any location!');
    return path.join(process.cwd(), 'python', 'process_csv_hybrid.py'); // Default path
  }
  
  /**
   * Process a CSV file using the hybrid model
   * @param {string} filePath - Path to the CSV file
   * @param {string} textColumn - Name of the column containing text data
   * @param {boolean} validate - Whether to validate with ground truth data
   * @returns {Promise<object>} - Processing results and metrics
   */
  async processCSV(filePath, textColumn = 'text', validate = false) {
    const sessionId = nanoid(10);
    
    return new Promise((resolve, reject) => {
      console.log(`🧠 Starting hybrid model processing for file: ${filePath}`);
      console.log(`Using Python binary: ${this.pythonBinary}`);
      
      // Prepare output path
      const outputPath = `${filePath.replace('.csv', '')}_hybrid_results.json`;
      
      // Create processing arguments
      const args = [
        this.scriptPath,
        '--input', filePath,
        '--output', outputPath,
        '--text_column', textColumn,
      ];
      
      if (validate) {
        args.push('--validate');
      }
      
      // Spawn the Python process
      console.log(`Running command: ${this.pythonBinary} ${args.join(' ')}`);
      const process = spawn(this.pythonBinary, args, {
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
      });
      
      // Store the process for potential cancellation
      this.activeProcesses.set(sessionId, {
        process,
        filePath,
        startTime: new Date(),
        progress: { processed: 0, total: 0, stage: 'Starting' }
      });
      
      // Collect stdout
      let stdoutData = '';
      process.stdout.on('data', (data) => {
        stdoutData += data.toString();
      });
      
      // Monitor stderr for progress updates and errors
      process.stderr.on('data', (data) => {
        const dataStr = data.toString();
        console.log(`Hybrid model stderr: ${dataStr}`);
        
        // Parse progress updates
        const progressRegex = /PROGRESS:(.+?)::END_PROGRESS/g;
        let match;
        
        // Find all progress updates in the output
        while ((match = progressRegex.exec(dataStr)) !== null) {
          try {
            const progressData = JSON.parse(match[1]);
            
            // Update progress tracking
            const processInfo = this.activeProcesses.get(sessionId);
            if (processInfo) {
              processInfo.progress = progressData;
              this.activeProcesses.set(sessionId, processInfo);
            }
            
            // Broadcast progress update via pythonService
            pythonService.broadcastProgress(sessionId, progressData);
          } catch (err) {
            console.error(`Error parsing progress data: ${err.message}`);
          }
        }
      });
      
      // Handle process completion
      process.on('close', (code) => {
        this.activeProcesses.delete(sessionId);
        
        if (code === 0) {
          // Success - read the output file
          try {
            if (fs.existsSync(outputPath)) {
              const resultData = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
              console.log(`✅ Hybrid model processing completed successfully`);
              
              // If metrics are available, log summary
              if (resultData.metrics) {
                console.log(`Model accuracy: ${resultData.metrics.accuracy}`);
                console.log(`F1 score (weighted): ${resultData.metrics.f1_score.weighted}`);
              }
              
              resolve({
                success: true,
                sessionId,
                results: resultData.results,
                metrics: resultData.metrics || null,
                processingTime: resultData.processing_time,
                totalProcessed: resultData.total_processed
              });
            } else {
              console.error(`❌ Output file not found: ${outputPath}`);
              reject(new Error(`Processing completed, but output file not found: ${outputPath}`));
            }
          } catch (err) {
            console.error(`❌ Error reading output file: ${err.message}`);
            reject(err);
          }
        } else {
          console.error(`❌ Hybrid model processing failed with code ${code}`);
          reject(new Error(`Processing failed with code ${code}`));
        }
      });
      
      // Handle process errors
      process.on('error', (err) => {
        this.activeProcesses.delete(sessionId);
        console.error(`❌ Error executing hybrid model process: ${err.message}`);
        reject(err);
      });
    });
  }
  
  /**
   * Cancel processing for a session
   * @param {string} sessionId - Session ID to cancel
   * @returns {boolean} - Whether cancellation was successful
   */
  cancelProcessing(sessionId) {
    const processInfo = this.activeProcesses.get(sessionId);
    if (processInfo) {
      console.log(`🛑 Canceling hybrid model processing for session: ${sessionId}`);
      
      try {
        processInfo.process.kill();
        this.activeProcesses.delete(sessionId);
        return true;
      } catch (err) {
        console.error(`Error canceling process: ${err.message}`);
        return false;
      }
    }
    
    return false;
  }
  
  /**
   * Get active processing sessions
   * @returns {Array} List of active session IDs
   */
  getActiveSessions() {
    return Array.from(this.activeProcesses.keys());
  }
  
  /**
   * Get detailed information about active processing sessions
   * @returns {Array} List of session information objects
   */
  getActiveSessionsInfo() {
    return Array.from(this.activeProcesses.entries()).map(([sessionId, info]) => ({
      sessionId,
      filePath: info.filePath,
      startTime: info.startTime,
      progress: info.progress
    }));
  }
  
  /**
   * Get progress for a specific session
   * @param {string} sessionId - Session ID to get progress for
   * @returns {object|null} Progress information or null if session not found
   */
  getSessionProgress(sessionId) {
    const processInfo = this.activeProcesses.get(sessionId);
    if (processInfo) {
      return processInfo.progress;
    }
    return null;
  }
  
  /**
   * Cancel all active processing sessions
   * @returns {number} Number of sessions canceled
   */
  cancelAllSessions() {
    let canceledCount = 0;
    
    for (const sessionId of this.activeProcesses.keys()) {
      if (this.cancelProcessing(sessionId)) {
        canceledCount++;
      }
    }
    
    return canceledCount;
  }
}

// Create and export a singleton instance
export const hybridProcessor = new HybridModelProcessor();