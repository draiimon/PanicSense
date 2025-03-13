import { apiRequest } from './queryClient';

export interface SentimentPost {
  id: number;
  text: string;
  timestamp: string;
  source: string;
  language: string;
  sentiment: string;
  confidence: number;
  location: string | null;
  disasterType: string | null;
  fileId: number | null;
}

export interface DisasterEvent {
  id: number;
  name: string;
  description: string | null;
  timestamp: string;
  location: string | null;
  type: string;
  sentimentImpact: string | null;
}

export interface AnalyzedFile {
  id: number;
  originalName: string;
  storedName: string;
  timestamp: string;
  recordCount: number;
  evaluationMetrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    confusionMatrix: number[][];
  } | null;
}

export interface UploadProgress {
  processed: number;
  stage: string;
}

// Sentiment Posts API
export async function getSentimentPosts(): Promise<SentimentPost[]> {
  const response = await apiRequest('GET', '/api/sentiment-posts');
  return response.json();
}

export async function getSentimentPostsByFileId(fileId: number): Promise<SentimentPost[]> {
  const response = await apiRequest('GET', `/api/sentiment-posts/file/${fileId}`);
  return response.json();
}

// Disaster Events API
export async function getDisasterEvents(): Promise<DisasterEvent[]> {
  const response = await apiRequest('GET', '/api/disaster-events');
  return response.json();
}

// Analyzed Files API
export async function getAnalyzedFiles(): Promise<AnalyzedFile[]> {
  const response = await apiRequest('GET', '/api/analyzed-files');
  return response.json();
}

export async function getAnalyzedFile(id: number): Promise<AnalyzedFile> {
  const response = await apiRequest('GET', `/api/analyzed-files/${id}`);
  return response.json();
}

// File Upload
export async function uploadCSV(
  file: File,
  onProgress?: (progress: UploadProgress) => void
): Promise<{
  file: AnalyzedFile;
  posts: SentimentPost[];
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  } | null;
}> {
  const formData = new FormData();
  formData.append('file', file);

  // Set up event source for progress updates
  const eventSource = new EventSource('/api/upload-progress');

  eventSource.onmessage = (event) => {
    const progress = JSON.parse(event.data);
    if (onProgress) {
      onProgress(progress);
    }
  };

  try {
    const response = await fetch('/api/upload-csv', {
      method: 'POST',
      body: formData,
      credentials: 'include',
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to upload CSV: ${errorText}`);
    }

    return response.json();
  } finally {
    eventSource.close();
  }
}

// Text Analysis
export async function analyzeText(text: string): Promise<{
  post: SentimentPost;
}> {
  const response = await apiRequest('POST', '/api/analyze-text', { text });
  return response.json();
}