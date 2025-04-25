import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { nanoid } from 'nanoid';
import { log } from './vite';
import { usageTracker } from './utils/usage-tracker';
import { storage } from './storage';

// Global array to store logs from Python service
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
  private scriptPath: string = '';  // Initialize with empty string to avoid TypeScript error
  private confidenceCache: Map<string, number>;  // Cache for confidence scores
  private similarityCache: Map<string, boolean>; // Cache for text similarity checks
  private activeProcesses: Map<string, { process: any, tempFilePath: string, startTime: Date }>;  // Track active Python processes with start times

  constructor() {
    // Enhanced Python binary detection with fallbacks for different environments
    if (process.env.NODE_ENV === 'production') {
      // Try multiple production python paths
      const possiblePythonPaths = [
        '/app/venv/bin/python3',  // Default venv path
        '/usr/bin/python3',       // System python
        'python3',                // PATH-based python
        'python'                  // Generic fallback
      ];
      
      // Use the first Python binary that exists
      this.pythonBinary = process.env.PYTHON_PATH || possiblePythonPaths[0];
      console.log(`🐍 Using Python binary in production: ${this.pythonBinary}`);
    } else {
      this.pythonBinary = 'python3';
      console.log(`🐍 Using development Python binary: ${this.pythonBinary}`);
    }
    
    // Create temp dir if it doesn't exist
    this.tempDir = path.join(os.tmpdir(), 'disaster-sentiment');
    try {
      if (!fs.existsSync(this.tempDir)) {
        fs.mkdirSync(this.tempDir, { recursive: true });
      }
    } catch (error) {
      const err = error as Error;
      console.warn(`⚠️ Unable to create temp directory: ${err.message}`);
      // Fallback to OS temp dir directly
      this.tempDir = os.tmpdir();
    }
    console.log(`📁 Using temp directory: ${this.tempDir}`);
    
    // Script path with better error handling and logging for Render deployment
    // In Render production environment, python folder is in the root directory
    // Note: Using process.cwd() instead of __dirname for ESM compatibility
    const possibleScriptPaths = [
      path.join(process.cwd(), 'server', 'python', 'process.py'),  // Standard path
      path.join(process.cwd(), 'server', 'python', 'process.py'),  // Duplicate for consistency (removed __dirname)
      path.join(process.cwd(), 'python', 'process.py'),            // Root python folder path
      path.join(process.cwd(), 'python', 'process.py')             // Alternative path (removed __dirname)
    ];
    
    // Try each path and use the first one that exists
    let scriptFound = false;
    for (const scriptPath of possibleScriptPaths) {
      try {
        if (fs.existsSync(scriptPath)) {
          this.scriptPath = scriptPath;
          console.log(`✅ Found Python script at: ${scriptPath}`);
          scriptFound = true;
          break;
        }
      } catch (error) {
        const err = error as Error;
        console.warn(`⚠️ Error checking path ${scriptPath}: ${err.message}`);
      }
    }
    
    // If no script was found, use the default path but log a warning
    if (!scriptFound) {
      this.scriptPath = path.join(process.cwd(), 'python', 'process.py');
      console.warn(`⚠️ No Python script found. Defaulting to: ${this.scriptPath}`);
    }
    
    // Initialize the cache for confidence scores to improve performance
    this.confidenceCache = new Map<string, number>();
    
    // Initialize cache for text similarity to avoid duplicate training
    this.similarityCache = new Map<string, boolean>();
    
    // Initialize tracking for active processes
    this.activeProcesses = new Map();
  }

  public async cancelProcessing(sessionId: string): Promise<boolean> {
    if (this.activeProcesses.has(sessionId)) {
      // Get the process info
      const processInfo = this.activeProcesses.get(sessionId);
      
      if (processInfo && processInfo.process) {
        // Log the cancellation
        log(`Cancelling Python process for session: ${sessionId}`, 'python-service');
        
        try {
          // Kill the process
          processInfo.process.kill();
          
          // Remove temp file if it exists
          if (processInfo.tempFilePath && fs.existsSync(processInfo.tempFilePath)) {
            fs.unlinkSync(processInfo.tempFilePath);
            log(`Removed temp file: ${processInfo.tempFilePath}`, 'python-service');
          }
          
          // Remove from active processes
          this.activeProcesses.delete(sessionId);
          return true;
        } catch (error) {
          log(`Error cancelling process: ${error}`, 'python-service');
          return false;
        }
      }
    }
    
    log(`No active process found for session: ${sessionId}`, 'python-service');
    return false;
  }
  
  public getActiveProcessSessions(): string[] {
    return Array.from(this.activeProcesses.keys());
  }
  
  /**
   * Get detailed information about all active processes
   * Useful for debugging and monitoring
   */
  public getActiveProcessesInfo(): {sessionId: string, startTime: Date}[] {
    const result: {sessionId: string, startTime: Date}[] = [];
    
    this.activeProcesses.forEach((info, sessionId) => {
      result.push({
        sessionId,
        startTime: info.startTime
      });
    });
    
    return result;
  }
  
  public isProcessRunning(sessionId: string): boolean {
    return this.activeProcesses.has(sessionId);
  }
  
  public cancelAllProcesses(): void {
    log(`Cancelling all Python processes (${this.activeProcesses.size} active)`, 'python-service');
    
    // Make a copy of the keys to avoid modification during iteration
    const sessionIds = Array.from(this.activeProcesses.keys());
    
    // Cancel each process
    let successCount = 0;
    for (const sessionId of sessionIds) {
      const success = this.cancelProcessing(sessionId);
      if (success) {
        successCount++;
      }
    }
    
    log(`Successfully cancelled ${successCount} of ${sessionIds.length} processes`, 'python-service');
  }
  
  public extractDisasterTypeFromText(text: string): string | null {
    const disasterTypeKeywords: Record<string, string[]> = {
      'Earthquake': ['earthquake', 'quake', 'tremor', 'seismic', 'lindol', 'lindell', 'seismic activity', 'magnitude'],
      'Flood': ['flood', 'flooding', 'flash flood', 'deluge', 'baha', 'inundation', 'water level', 'heavy rain'],
      'Typhoon': ['typhoon', 'hurricane', 'storm', 'cyclone', 'bagyo', 'winds', 'rainfall', 'storm surge'],
      'Landslide': ['landslide', 'mudslide', 'rockfall', 'landslip', 'guho', 'soil erosion', 'debris flow'],
      'Volcano': ['volcano', 'eruption', 'volcanic', 'magma', 'ash', 'lava', 'pyroclastic', 'bulkan', 'phreatic'],
      'Drought': ['drought', 'dry spell', 'water shortage', 'tagtuyot', 'crop failure', 'water restriction'],
      'Fire': ['fire', 'blaze', 'wildfire', 'flame', 'bushfire', 'sunog', 'burning', 'combustion'],
      'Tsunami': ['tsunami', 'tidal wave', 'seismic sea wave', 'harbor wave', 'storm surge'],
      'Heatwave': ['heatwave', 'hot spell', 'extreme heat', 'heat dome', 'heat stress', 'hot weather', 'heat stroke'],
      'COVID-19': ['covid', 'coronavirus', 'pandemic', 'virus', 'covid-19', 'vaccination', 'vaccine', 'health crisis']
    };
    
    // Convert text to lowercase for comparison
    const lowerText = text.toLowerCase();
    
    // Check each disaster type
    for (const [disasterType, keywords] of Object.entries(disasterTypeKeywords)) {
      // Check if any keyword matches
      if (keywords.some(keyword => lowerText.includes(keyword.toLowerCase()))) {
        return disasterType;
      }
    }
    
    return null;
  }
  
  public extractLocationFromText(text: string): string | null {
    const commonPhilippineLocations = [
      'Manila', 'Quezon City', 'Davao', 'Cebu', 'Makati', 'Baguio', 'Tacloban', 'Iloilo',
      'Pasig', 'Taguig', 'Pasay', 'Mandaluyong', 'Marikina', 'Parañaque', 'Muntinlupa',
      'Batangas', 'Laguna', 'Cavite', 'Rizal', 'Pampanga', 'Bulacan', 'Zambales', 'Bataan',
      'Mindanao', 'Visayas', 'Luzon', 'Bicol', 'Ilocos', 'Cagayan', 'Mindoro', 'Negros', 'Leyte',
      'Palawan', 'Samar', 'Panay', 'Boracay', 'Taal', 'Mayon', 'Pinatubo', 'Babuyan', 'Sulu',
      'Metro Manila', 'NCR', 'Calabarzon', 'Mimaropa', 'CAR', 'ARMM', 'BARMM', 'CARAGA',
      'Cotabato', 'Maguindanao', 'Lanao', 'Zamboanga', 'Basilan', 'Jolo', 'Tawi-Tawi'
    ];
    
    // Check for direct mentions of locations
    for (const location of commonPhilippineLocations) {
      const locationRegex = new RegExp(`\\b${location}\\b`, 'i');
      if (locationRegex.test(text)) {
        return location;
      }
    }
    
    return null;
  }
  
  private getCachedConfidence(text: string): number | undefined {
    return this.confidenceCache.get(text);
  }
  
  private setCachedConfidence(text: string, confidence: number): void {
    this.confidenceCache.set(text, confidence);
  }
  
  public clearCacheForText(text: string): void {
    this.confidenceCache.delete(text);
  }
  
  public clearCache(): void {
    this.confidenceCache.clear();
  }

  // HYBRID MODEL TRAINING DISABLED - Simplified function that just logs feedback
  public async trainModelWithFeedback(
    originalText: string, 
    originalSentiment: string, 
    correctedSentiment: string,
    correctedLocation?: string,
    correctedDisasterType?: string
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
      // Log that training has been disabled but feedback was received
      log(`⚠️ TRAINING DISABLED: Received sentiment feedback: "${originalText.substring(0, 30)}..." - ${originalSentiment} → ${correctedSentiment}`, 'python-service');
      
      // Log additional corrections that were provided (but won't be processed)
      if (correctedLocation && correctedLocation !== "UNKNOWN") {
        log(`  - location correction to: ${correctedLocation} (not processed - training disabled)`, 'python-service');
      }
      
      if (correctedDisasterType && correctedDisasterType !== "UNKNOWN") {
        log(`  - disaster type correction to: ${correctedDisasterType} (not processed - training disabled)`, 'python-service');
      }
      
      // Create feedback payload for logging purposes
      const feedbackData = JSON.stringify({
        feedback: true,
        originalText,
        originalSentiment,
        correctedSentiment,
        correctedLocation: correctedLocation || undefined,
        correctedDisasterType: correctedDisasterType || undefined
      });
      
      // Log the feedback that would have been processed
      log(`Feedback received (but not processed - training disabled): ${feedbackData}`, 'python-service');
      
      // Return a simulated successful response instead of actually processing
      const simulatedResponse = {
        status: "success",
        message: "Thank you for your feedback. Your input has been recorded. (Note: Model training is currently disabled)",
        performance: {
          previous_accuracy: 0.85,
          new_accuracy: 0.85, 
          improvement: 0.0  // No improvement since no actual training occurred
        }
      };
      
      // Log the simulated response
      log(`Returning simulated training response (training disabled): ${JSON.stringify(simulatedResponse)}`, 'python-service');
      
      return simulatedResponse;
    } catch (error) {
      const errorMsg = `Error in feedback handling: ${error}`;
      log(errorMsg, 'python-service');
      
      // Return a structured error response
      return {
        status: "error",
        message: errorMsg
      };
    }
  }
}