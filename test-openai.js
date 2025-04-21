/**
 * Simple OpenAI API test script
 * This script tests if the OpenAI API key is working correctly
 */

import { OpenAI } from 'openai';
import { fileURLToPath } from 'url';
import path from 'path';

// Initialize OpenAI client with API key from environment variable
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

async function testOpenAI() {
  console.log('Testing OpenAI API connection...');

  try {
    // Simple API test - requesting a short message
    const response = await openai.chat.completions.create({
      model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello from PanicSense! Can you confirm the OpenAI API is working?" }
      ],
      max_tokens: 100,
    });

    // Check if we got a valid response
    if (response && response.choices && response.choices.length > 0) {
      console.log('✅ OpenAI API connection successful!');
      console.log('Response:', response.choices[0].message.content);
      return true;
    } else {
      console.error('❌ OpenAI API returned an empty response');
      console.error('Response object:', response);
      return false;
    }
  } catch (error) {
    console.error('❌ OpenAI API Error:', error.message);
    if (error.response) {
      console.error('Error details:', error.response.data);
    }
    return false;
  }
}

// Execute the test if this is the main module
const isMainModule = process.argv[1] === fileURLToPath(import.meta.url);
if (isMainModule) {
  testOpenAI().then(success => {
    if (success) {
      console.log('✅ API test completed successfully');
    } else {
      console.error('❌ API test failed');
      process.exit(1);
    }
  }).catch(err => {
    console.error('Unexpected error:', err);
    process.exit(1);
  });
}

// Export for use in other files
export { testOpenAI };