/**
 * Test script for disaster detection
 * This script tests if the disaster detection functionality is working
 */

import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import os from 'os';

// Define a simple news item to test with
const testNewsItem = {
  title: "Flood warnings issued for Manila after heavy rainfall",
  content: "MANILA - Several areas in Metro Manila are on high alert after continuous heavy rainfall has caused flooding in low-lying areas. The PAGASA has issued warnings for residents along riverbanks and flood-prone areas to evacuate. Emergency services are on standby as water levels continue to rise."
};

// Temporary file paths
const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'disaster-test-'));
const inputFilePath = path.join(tempDir, 'test-input.json');
const outputFilePath = path.join(tempDir, 'test-output.json');

// Write test data to temporary file
fs.writeFileSync(inputFilePath, JSON.stringify(testNewsItem));

/**
 * Test the disaster analysis functionality
 */
async function testDisasterAnalysis() {
  console.log('Testing disaster analysis functionality...');
  console.log('Test news item:', testNewsItem.title);
  
  try {
    // Find Python script path
    const scriptPath = path.resolve(process.cwd(), 'python', 'process.py');
    
    if (!fs.existsSync(scriptPath)) {
      console.error(`❌ Python script not found at: ${scriptPath}`);
      return false;
    }
    
    console.log(`Using Python script at: ${scriptPath}`);
    
    // Spawn Python process
    const pythonProcess = spawn('python3', [
      scriptPath,
      inputFilePath
    ]);
    
    // Collect output
    let outputData = '';
    let errorData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
    });
    
    // Wait for process to complete
    const exitCode = await new Promise((resolve) => {
      pythonProcess.on('close', resolve);
    });
    
    // Check for success
    if (exitCode === 0 && outputData) {
      console.log('✅ Disaster analysis process completed successfully');
      
      try {
        // Try to parse the output as JSON
        const result = JSON.parse(outputData);
        console.log('Analysis result:', JSON.stringify(result, null, 2));
        
        if (result.is_disaster_related !== undefined) {
          console.log(`Is disaster related: ${result.is_disaster_related}`);
          console.log(`Disaster type: ${result.disaster_type || 'Not specified'}`);
          console.log(`Location: ${result.location || 'Not specified'}`);
          console.log(`Confidence: ${result.confidence || 0}`);
          return true;
        } else {
          console.error('❌ Invalid analysis result format');
          return false;
        }
      } catch (parseError) {
        console.error('❌ Error parsing analysis result:', parseError.message);
        console.error('Raw output:', outputData);
        return false;
      }
    } else {
      console.error(`❌ Disaster analysis process failed with exit code: ${exitCode}`);
      if (errorData) {
        console.error('Error output:', errorData);
      }
      return false;
    }
  } catch (error) {
    console.error('❌ Error during disaster analysis test:', error.message);
    return false;
  } finally {
    // Clean up temporary files
    try {
      fs.unlinkSync(inputFilePath);
    } catch (e) {
      // Ignore cleanup errors
    }
  }
}

// Run the test
testDisasterAnalysis().then(success => {
  if (success) {
    console.log('✅ Disaster analysis test completed successfully');
  } else {
    console.error('❌ Disaster analysis test failed');
    process.exit(1);
  }
}).catch(err => {
  console.error('Unexpected error:', err);
  process.exit(1);
});