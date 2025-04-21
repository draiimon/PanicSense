/**
 * News Validator using Groq API
 * This is a separate tool for validating if news content is related to a legitimate disaster
 */

const axios = require('axios');

// Use the provided Groq API key for validation
const GROQ_API_KEY = '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';

/**
 * Validates if a news article is describing a legitimate disaster
 * @param {string} title - The news article title
 * @param {string} content - The news article content
 * @returns {Promise<{isDisaster: boolean, confidence: number, disasterType: string|null, details: string}>}
 */
async function validateNewsContent(title, content) {
  try {
    const validationPrompt = `
You are a disaster news validation assistant. Your task is to analyze the provided news article and determine if it is describing a legitimate disaster.

News title: ${title}
News content: ${content}

Please analyze the content and determine:
1. Is this describing a legitimate disaster or emergency event? (Yes/No)
2. What type of disaster is it? (e.g., Flood, Earthquake, Fire, Typhoon, etc. or None if not a disaster)
3. How confident are you about this classification (a number between 0 and 1)
4. Provide a brief explanation for your decision

Format your response as a valid JSON object with the following fields:
{
  "isDisaster": boolean,
  "disasterType": string or null,
  "confidence": number between 0 and 1,
  "details": explanation as string
}
`;

    const response = await axios.post(
      'https://api.groq.com/openai/v1/chat/completions',
      {
        model: 'llama-3.1-8b-instant',
        messages: [
          {
            role: 'system',
            content: 'You are a specialized disaster news validation tool. Respond only with valid JSON.'
          },
          {
            role: 'user',
            content: validationPrompt
          }
        ],
        max_tokens: 1000,
        temperature: 0.2,
        top_p: 0.9,
        stream: false
      },
      {
        headers: {
          'Authorization': `Bearer ${GROQ_API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    );

    try {
      // Extract the JSON from the response
      const responseText = response.data.choices[0].message.content;
      
      // Parse the JSON
      const jsonResponse = JSON.parse(responseText);
      
      // Validate the response has the expected fields
      if (typeof jsonResponse.isDisaster !== 'boolean' || 
          typeof jsonResponse.confidence !== 'number' ||
          typeof jsonResponse.details !== 'string') {
        throw new Error('Invalid response format');
      }
      
      return {
        isDisaster: jsonResponse.isDisaster,
        disasterType: jsonResponse.disasterType || null,
        confidence: jsonResponse.confidence,
        details: jsonResponse.details
      };
    } catch (parseError) {
      console.error('Error parsing AI response:', parseError);
      // Fallback manual extraction logic
      const responseText = response.data.choices[0].message.content;
      
      // Very basic fallback extraction using regex
      const isDisaster = /("isDisaster"\s*:\s*true|"isDisaster"\s*:\s*1)/i.test(responseText);
      const confidenceMatch = responseText.match(/"confidence"\s*:\s*([0-9.]+)/i);
      const confidence = confidenceMatch ? parseFloat(confidenceMatch[1]) : 0.5;
      const disasterTypeMatch = responseText.match(/"disasterType"\s*:\s*"([^"]*)"/i);
      const disasterType = disasterTypeMatch ? disasterTypeMatch[1] : null;
      const detailsMatch = responseText.match(/"details"\s*:\s*"([^"]*)"/i);
      const details = detailsMatch ? detailsMatch[1] : 'No explanation provided';
      
      return {
        isDisaster,
        disasterType,
        confidence,
        details
      };
    }
  } catch (error) {
    console.error('Error validating news with Groq API:', error.message);
    // Default fallback response
    return {
      isDisaster: false,
      disasterType: null,
      confidence: 0,
      details: `Error during validation: ${error.message}`
    };
  }
}

module.exports = {
  validateNewsContent
};