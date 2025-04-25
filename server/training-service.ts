/**
 * Training Service Integration for PanicSense
 * Integrates with Python-based sentiment analysis training
 */

import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { db } from './db';
import * as schema from '../shared/schema';
import { and, eq } from 'drizzle-orm';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';

// ES Module fix for __dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Constants for file paths
const PYTHON_SCRIPT_PATH = path.join(__dirname, '..', 'python', 'sentiment_trainer.py');
const TEMP_DIR = path.join(__dirname, '..', 'uploads');
const MODELS_DIR = path.join(__dirname, '..', 'python', 'models');

// Ensure directories exist
if (!fs.existsSync(TEMP_DIR)) {
  fs.mkdirSync(TEMP_DIR, { recursive: true });
}
if (!fs.existsSync(MODELS_DIR)) {
  fs.mkdirSync(MODELS_DIR, { recursive: true });
}

/**
 * Interface for evaluation metrics
 */
export interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: number[][];
  labels: string[];
  trainingDate: string;
  testSize: number;
  sampleCount: number;
}

/**
 * Generate a random CSV file with sample data for testing
 */
async function generateSampleData(numSamples: number = 50): Promise<string> {
  console.log(`Generating sample data with ${numSamples} samples`);
  
  const sentiments = ["panic", "fear", "disbelief", "resilience", "neutral"];
  const exampleTexts = {
    panic: [
      "Help! The earthquake destroyed our house!",
      "We're going to die in this flood!",
      "Run! The fire is spreading too fast!",
      "SOS! We're trapped in the building!",
      "Oh my god! It's flooding everywhere! We're trapped!"
    ],
    fear: [
      "I'm scared of the aftershocks.",
      "This earthquake is scary. I'm worried about aftershocks.",
      "I don't feel safe in this building anymore.",
      "I'm afraid for my family's safety during this storm.",
      "The water is rising, I'm getting worried."
    ],
    disbelief: [
      "I can't believe this is happening to us.",
      "Did that really just happen? I can't believe it.",
      "Is this real? I've never seen flooding this bad before.",
      "This can't be happening. The whole town is underwater?",
      "Did the typhoon really destroy the entire barangay?"
    ],
    resilience: [
      "We will rebuild our community together.",
      "I hope everyone stays safe during this typhoon. 🙏",
      "Together, we can overcome this disaster. Stay strong!",
      "Praying for everyone's safety. God bless us all.",
      "No matter what happens, we will get through this together."
    ],
    neutral: [
      "The typhoon is expected to make landfall tomorrow morning.",
      "PAGASA reports the typhoon will make landfall at 2pm today.",
      "The governor announced relief operations will begin tomorrow.",
      "Relief goods will be distributed at the evacuation center.",
      "The bridge is closed due to flooding."
    ]
  };
  
  let csvContent = "text,sentiment\n";
  
  for (let i = 0; i < numSamples; i++) {
    // Select a random sentiment
    const sentiment = sentiments[Math.floor(Math.random() * sentiments.length)];
    
    // Select a random example for that sentiment
    const examples = exampleTexts[sentiment as keyof typeof exampleTexts];
    const text = examples[Math.floor(Math.random() * examples.length)];
    
    // Escape any commas in the text
    const escapedText = text.replace(/"/g, '""');
    
    // Add to CSV
    csvContent += `"${escapedText}",${sentiment}\n`;
  }
  
  // Create a temporary file
  const filePath = path.join(TEMP_DIR, `sample_data_${uuidv4()}.csv`);
  fs.writeFileSync(filePath, csvContent);
  
  return filePath;
}

/**
 * Train a model with a CSV file and save evaluation metrics
 */
export async function trainWithCSV(
  filePath: string, 
  fileId: number,
  onProgress?: (progress: any) => void
): Promise<{
  metrics: EvaluationMetrics,
  success: boolean,
  message: string
}> {
  return new Promise((resolve, reject) => {
    console.log(`Training model with CSV file: ${filePath}`);
    
    // Validate file exists
    if (!fs.existsSync(filePath)) {
      return reject(new Error(`File not found: ${filePath}`));
    }
    
    // Create Python process
    const pythonProcess = spawn('python3', [PYTHON_SCRIPT_PATH, '--train', filePath]);
    
    let outputData = '';
    let errorData = '';
    
    // Collect stdout data
    pythonProcess.stdout.on('data', (data) => {
      const chunk = data.toString();
      outputData += chunk;
      
      // Report progress if callback provided
      if (onProgress && chunk.includes('progress:')) {
        try {
          const progressMatch = chunk.match(/progress:\s*(\d+)/);
          if (progressMatch && progressMatch[1]) {
            const progressValue = parseInt(progressMatch[1], 10);
            onProgress({
              processed: progressValue,
              total: 100,
              stage: 'Training model',
              timestamp: Date.now()
            });
          }
        } catch (err) {
          console.error('Error parsing progress:', err);
        }
      }
    });
    
    // Collect stderr data
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error(`Python error: ${data.toString()}`);
    });
    
    // Handle process completion
    pythonProcess.on('close', async (code) => {
      if (code !== 0) {
        console.error(`Python process exited with code ${code}`);
        console.error(`Error: ${errorData}`);
        return reject(new Error(`Python process failed: ${errorData}`));
      }
      
      try {
        // Find the JSON metrics in the output
        const jsonMatch = outputData.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          return reject(new Error('No metrics found in Python output'));
        }
        
        // Parse metrics JSON
        const metrics: EvaluationMetrics = JSON.parse(jsonMatch[0]);
        
        // Update the file record with metrics
        await db
          .update(schema.analyzedFiles)
          .set({
            evaluationMetrics: metrics as any
          })
          .where(eq(schema.analyzedFiles.id, fileId));
        
        resolve({
          metrics,
          success: true,
          message: 'Model trained successfully'
        });
      } catch (err) {
        console.error('Error processing Python output:', err);
        reject(err);
      }
    });
  });
}

/**
 * Create a demo dataset and train a model
 */
export async function createDemoDataAndTrain(
  recordCount: number = 100,
  onProgress?: (progress: any) => void
): Promise<{
  fileId: number;
  metrics: EvaluationMetrics;
  success: boolean;
  message: string;
}> {
  try {
    // Generate a filename for the stored file
    const storedFilename = `demo_dataset_${Date.now()}.csv`;
    
    // First create a file record in the database
    const [fileRecord] = await db
      .insert(schema.analyzedFiles)
      .values({
        originalName: 'Demo Dataset',
        storedName: storedFilename,
        recordCount: recordCount,
        evaluationMetrics: null
      })
      .returning();
    
    // Generate sample data file
    const filePath = await generateSampleData(recordCount);
    
    // Train model with the file
    const { metrics, success, message } = await trainWithCSV(
      filePath, 
      fileRecord.id,
      onProgress
    );
    
    // Copy the generated file to the uploads directory with the stored filename
    const destinationPath = path.join(TEMP_DIR, storedFilename);
    fs.copyFileSync(filePath, destinationPath);
    
    // Clean up temporary file
    fs.unlinkSync(filePath);
    
    // Generate sentiment posts from the sample data
    await generateSentimentPostsFromFile(destinationPath, fileRecord.id);
    
    return {
      fileId: fileRecord.id,
      metrics,
      success,
      message
    };
  } catch (err) {
    console.error('Error creating demo data and training:', err);
    throw err;
  }
}

/**
 * Generate sentiment posts from a CSV file
 */
async function generateSentimentPostsFromFile(filePath: string, fileId: number): Promise<void> {
  try {
    // Read the CSV file
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const lines = fileContent.split('\n');
    
    // Skip header line
    const dataLines = lines.slice(1);
    
    // Process each line
    for (const line of dataLines) {
      if (!line.trim()) continue;
      
      // Parse CSV line (handling quoted fields with commas)
      let text = '';
      let sentiment = '';
      
      // Simple CSV parsing
      if (line.includes('"')) {
        // Has quoted fields
        const quoteMatch = line.match(/"([^"]*)",(.*)/);
        if (quoteMatch) {
          text = quoteMatch[1].replace(/""/g, '"');
          sentiment = quoteMatch[2].trim();
        }
      } else {
        // Simple case
        const parts = line.split(',');
        text = parts[0].trim();
        sentiment = parts[1]?.trim() || '';
      }
      
      if (!text || !sentiment) continue;
      
      // Create sentiment post
      await db.insert(schema.sentimentPosts).values({
        text,
        sentiment,
        confidence: 0.8 + Math.random() * 0.2, // Random high confidence
        source: 'demo',
        language: 'english',
        fileId
      });
    }
    
    console.log(`Generated sentiment posts from file: ${filePath}`);
  } catch (err) {
    console.error('Error generating sentiment posts:', err);
    throw err;
  }
}

/**
 * Main class for training service
 */
export class TrainingService {
  /**
   * Train a model with a CSV file
   */
  async trainWithFile(
    filePath: string, 
    fileId: number,
    onProgress?: (progress: any) => void
  ): Promise<{
    metrics: EvaluationMetrics;
    success: boolean;
    message: string;
  }> {
    return trainWithCSV(filePath, fileId, onProgress);
  }
  
  /**
   * Create a demo dataset and train a model
   */
  async createDemoDataset(
    recordCount: number = 100,
    onProgress?: (progress: any) => void
  ): Promise<{
    fileId: number;
    metrics: EvaluationMetrics;
    success: boolean;
    message: string;
  }> {
    return createDemoDataAndTrain(recordCount, onProgress);
  }
}

// Export singleton instance
export const trainingService = new TrainingService();