/**
 * Test script to simulate CSV upload with ISO 8601 timestamps
 */
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// Create a small CSV file with ISO 8601 timestamps
const createTestCsv = () => {
  const csvContent = `text,cleaned_text,date,source,location,disaster_type
Well pray for your fast recovery. I hope that the spirit of Christmas still lives on despite the calamity. Godbless to all of you.,well pray for your fast recovery i hope that the spirit of christmas still lives on despite the calamity godbless to all of you,2019-12-25T04:27:04.000Z,Facebook,Capiz,Typhoon
"Lord I pray that you touch them with your hands of safety and guidance to all those regions under flood conditions. Bless these people Lord and keep them safe. Do not let the flood waters steal their lives. Please drive back the floods to protect their homes and their love ones.",lord i pray that you touch them with your hands of safety and guidance to all those regions under flood conditions bless these people lord and keep them safe do not let the flood waters steal their lives please drive back the floods to protect their homes and their love ones,2019-12-25T12:44:40.000Z,Facebook,Capiz,Typhoon
`;

  const testCsvPath = path.join(__dirname, 'test-sample.csv');
  fs.writeFileSync(testCsvPath, csvContent);
  console.log(`Created test CSV at ${testCsvPath}`);
  return testCsvPath;
};

// Run the Python script directly with the test CSV
const runPythonTest = (csvPath) => {
  console.log('Running Python processing script with test CSV...');
  
  const pythonPath = process.platform === 'win32' ? 'python' : 'python3';
  const scriptPath = path.join(__dirname, '..', 'server', 'python', 'process.py');
  
  const pythonProcess = spawn(pythonPath, [
    scriptPath,
    '--file', csvPath
  ]);
  
  pythonProcess.stdout.on('data', (data) => {
    try {
      const output = data.toString();
      if (output.trim()) {
        const jsonResult = JSON.parse(output);
        console.log('🟢 Successfully processed CSV file!');
        console.log('Results:', JSON.stringify(jsonResult, null, 2));
        
        // Verify timestamp parsing
        if (jsonResult.results && jsonResult.results.length > 0) {
          console.log('\n🔍 Verifying timestamp processing:');
          jsonResult.results.forEach((result, i) => {
            console.log(`Item ${i+1}:`);
            console.log(`  Original timestamp: ${result.timestamp}`);
            
            // Parse timestamp to verify format
            try {
              const date = new Date(result.timestamp);
              console.log(`  Parsed as JavaScript Date: ${date}`);
              console.log(`  ISO format: ${date.toISOString()}`);
              console.log('  ✅ Successfully parsed timestamp');
            } catch (err) {
              console.log(`  ❌ Failed to parse timestamp: ${err.message}`);
            }
          });
        } else {
          console.log('❌ No results found in output');
        }
      }
    } catch (err) {
      console.log('Received non-JSON output:', data.toString());
    }
  });
  
  pythonProcess.stderr.on('data', (data) => {
    const output = data.toString();
    // Skip progress reports
    if (!output.includes('PROGRESS:')) {
      console.log('Python stderr:', output);
    }
  });
  
  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
    
    // Clean up test file
    fs.unlinkSync(csvPath);
    console.log(`Deleted test CSV file: ${csvPath}`);
  });
};

// Run the test
const testCsvPath = createTestCsv();
runPythonTest(testCsvPath);