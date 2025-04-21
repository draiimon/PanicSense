/**
 * Basic server test file for CI/CD pipeline
 */

// Simple test to verify that the health endpoint works
async function testHealthEndpoint() {
  try {
    const response = await fetch('http://localhost:5000/api/health');
    const data = await response.json();
    
    if (data.status === 'ok') {
      console.log('✅ Health endpoint test passed');
      return true;
    } else {
      console.error('❌ Health endpoint test failed: Unexpected response', data);
      return false;
    }
  } catch (error) {
    console.error('❌ Health endpoint test failed:', error.message);
    return false;
  }
}

// Run all tests
async function runTests() {
  const testResults = await Promise.all([
    testHealthEndpoint()
  ]);
  
  const passed = testResults.every(result => result === true);
  
  if (passed) {
    console.log('🎉 All tests passed!');
    process.exit(0);
  } else {
    console.error('❌ Some tests failed!');
    process.exit(1);
  }
}

// If this file is run directly, execute the tests
if (require.main === module) {
  runTests();
}

export { testHealthEndpoint, runTests };