/**
 * Type definitions for fix-taglish.js
 */

interface SentimentPostResult {
  text: string;
  timestamp: string;
  source: string;
  language: string;
  sentiment: string;
  confidence: number;
  explanation?: string;
  disasterType?: string;
  location?: string;
}

export function preserveTaglishEntries(results: SentimentPostResult[]): SentimentPostResult[];