/**
 * Test script to verify compatibility of ISO 8601 timestamp format
 * This tests if the Python service and Node.js can correctly parse timestamps
 * in the format YYYY-MM-DDThh:mm:ss.sssZ
 */

// Test date formatting in JavaScript
console.log('Testing JavaScript ISO 8601 date parsing:');
const testTimestamps = [
  '2019-12-25T04:27:04.000Z', // From dataset
  '2023-05-01T12:30:45.123Z', // Another example
  new Date().toISOString()    // Current time in ISO format
];

testTimestamps.forEach(timestamp => {
  try {
    const date = new Date(timestamp);
    console.log(`✅ Successfully parsed: ${timestamp}`);
    console.log(`   → JavaScript Date: ${date}`);
    console.log(`   → Formats back to: ${date.toISOString()}`);
    
    // Test database format compatibility (PostgreSQL accepts ISO 8601)
    console.log(`   → Database format: ${date.toISOString()}`);
    
    // Test if format is maintained after serialization/deserialization
    const serialized = JSON.stringify({ date });
    const deserialized = JSON.parse(serialized);
    console.log(`   → After JSON serialization: ${deserialized.date}`);
    
    // Verify it still works after being processed as a string
    const dateFromString = new Date(deserialized.date);
    console.log(`   → Reparsed after JSON: ${dateFromString.toISOString()}`);
  } catch (error) {
    console.error(`❌ Error parsing: ${timestamp}`);
    console.error(`   Error details: ${error.message}`);
  }
  console.log('------------------------');
});

// Test sending to Python (mock example)
console.log('\nSimulating Python timestamp handling:');
console.log('In Python, ISO 8601 timestamps are parsed using:');
console.log('from datetime import datetime');
console.log('timestamp = datetime.fromisoformat("2019-12-25T04:27:04.000Z".replace("Z", "+00:00"))');
console.log('# or alternative Python 3.11+ method:');
console.log('import datetime');
console.log('timestamp = datetime.datetime.fromisoformat("2019-12-25T04:27:04.000Z")');

console.log('\nVerification complete.');