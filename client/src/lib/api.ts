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
  explanation?: string | null;
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
  total?: number;
  stage: string;
  error?: string;
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

// File Upload with session handling
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

  // Generate a unique session ID
  const sessionId = crypto.randomUUID();

  // Set up event source for progress updates
  const eventSource = new EventSource(`/api/upload-progress/${sessionId}`);

  eventSource.onmessage = (event) => {
    try {
      const progress = JSON.parse(event.data);
      console.log('Progress event received:', progress);
      
      if (onProgress) {
        // Direct console log to see exactly what we're getting
        console.log('DEBUG PROGRESS VALUES:', {
          processed: progress.processed,
          total: progress.total,
          stage: progress.stage
        });
        
        // CRITICAL FIX: Force number conversion with Number() instead of parseInt
        // parseInt can fail with non-integer strings
        const processedNum = Number(progress.processed) || 0;
        const totalNum = Number(progress.total) || 100;
        
        // Create a safe progress object with numerical values
        const safeProgress = {
          processed: processedNum,
          total: totalNum,
          stage: progress.stage || 'Processing...',
          error: progress.error
        };
        
        console.log('Progress being sent to UI:', safeProgress);
        
        // CRITICAL DEBUG: Add a direct update with log to catch any issues
        console.log('DIRECTLY CALLING onProgress with:', safeProgress);
        
        // Call the onProgress callback
        onProgress(safeProgress);
        
        // Manually check if the DOM would update
        console.log('After calling onProgress:', safeProgress);
      }
    } catch (error) {
      console.error('Error parsing progress data:', error);
    }
  };

  try {
    const response = await fetch('/api/upload-csv', {
      method: 'POST',
      headers: {
        'X-Session-ID': sessionId
      },
      body: formData,
      credentials: 'include',
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to upload CSV');
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

// Delete Specific Sentiment Post
export async function deleteSentimentPost(id: number): Promise<{
  success: boolean;
  message: string;
}> {
  const response = await apiRequest('DELETE', `/api/sentiment-posts/${id}`);
  return response.json();
}

// Delete All Data
export async function deleteAllData(): Promise<{
  success: boolean;
  message: string;
}> {
  const response = await apiRequest('DELETE', '/api/delete-all-data');
  return response.json();
}