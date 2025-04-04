import { getUploadSessionId, clearUploadState } from '@/lib/upload-persistence';
import { queryClient } from '@/lib/queryClient';

// Progress callback type
export type ProgressCallback = (progress: any) => void;

// Main API request function with support for query params
export async function apiRequest<T = any>(
  endpoint: string,
  options: RequestInit = {},
  queryParams: Record<string, string> = {}
): Promise<T> {
  try {
    // Add query parameters if provided
    if (Object.keys(queryParams).length > 0) {
      const params = new URLSearchParams(queryParams);
      endpoint = `${endpoint}${endpoint.includes('?') ? '&' : '?'}${params.toString()}`;
    }

    const response = await fetch(endpoint, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      // Try to parse error message from the response
      try {
        const errorData = await response.json();
        throw new Error(errorData.message || `API request failed with status ${response.status}`);
      } catch (e) {
        // If parsing fails, throw a generic error with the status
        throw new Error(`API request failed with status ${response.status}`);
      }
    }

    // For 204 No Content, return empty object
    if (response.status === 204) {
      return {} as T;
    }

    // Check response content
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      return await response.json() as T;
    } else {
      // Return non-JSON responses as-is
      return await response.text() as unknown as T;
    }
  } catch (error) {
    console.error(`API request error for ${endpoint}:`, error);
    throw error;
  }
}

// Type Interfaces
export interface SentimentPost {
  id: number;
  text: string;
  timestamp: string;
  source: string;
  language: string;
  sentiment: string;
  confidence: number;
  disasterType?: string;
  location?: string;
  fileId?: number;
  explanation?: string;
}

export interface DisasterEvent {
  id: number;
  name: string;
  type: string;
  location: string;
  startDate: string;
  endDate?: string;
  affectedPeople?: number;
  economicLoss?: number;
}

export interface AnalyzedFile {
  id: number;
  fileName: string;
  originalName: string;
  fileSize: number;
  recordCount: number;
  uploadDate: string;
  processingTime: number;
  metrics?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
  };
}

// API Functions

// Check for active upload sessions
export async function checkForActiveSessions(): Promise<string | null> {
  try {
    console.log('Checking for active upload sessions...');
    const response = await apiRequest<{sessionId: string | null}>('/api/active-upload-session');
    console.log('Active upload session check complete:', response);
    return response.sessionId;
  } catch (error) {
    console.error('Error checking for active upload sessions:', error);
    return null;
  }
}

// Get current upload session ID from localStorage
export function getCurrentUploadSessionId(): string | null {
  return getUploadSessionId();
}

// Sentiment Posts API
export async function getSentimentPosts(): Promise<SentimentPost[]> {
  return apiRequest<SentimentPost[]>('/api/sentiment-posts');
}

export async function getSentimentPostsByFileId(fileId: number): Promise<SentimentPost[]> {
  return apiRequest<SentimentPost[]>(`/api/sentiment-posts/file/${fileId}`);
}

// Disaster Events API
export async function getDisasterEvents(): Promise<DisasterEvent[]> {
  return apiRequest<DisasterEvent[]>('/api/disaster-events');
}

// Analyzed Files API
export async function getAnalyzedFiles(): Promise<AnalyzedFile[]> {
  return apiRequest<AnalyzedFile[]>('/api/analyzed-files');
}

export async function getAnalyzedFile(id: number): Promise<AnalyzedFile> {
  return apiRequest<AnalyzedFile>(`/api/analyzed-files/${id}`);
}

// Upload CSV with progress tracking
export async function uploadCSV(
  file: File, 
  onProgress?: ProgressCallback
): Promise<{ file: AnalyzedFile, posts: SentimentPost[] } | undefined> {
  const formData = new FormData();
  formData.append('file', file);

  // Create and store session ID
  // Server will generate a session ID and return it
  
  try {
    const response = await fetch('/api/upload-csv', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || `Upload failed with status ${response.status}`);
    }

    const data = await response.json();
    
    // If we have a sessionId in the response, use it for progress tracking
    if (data.sessionId) {
      // Set up SSE connection for progress updates
      const eventSource = new EventSource(`/api/upload-progress/${data.sessionId}`);
      
      eventSource.onmessage = (event) => {
        try {
          const progress = JSON.parse(event.data);
          
          // Call the progress callback if provided
          if (onProgress) {
            onProgress(progress);
          }
          
          // If the upload is complete, error, or cancelled, close connection
          if (progress.stage && 
              (progress.stage.toLowerCase().includes('complete') || 
               progress.stage.toLowerCase().includes('error') ||
               progress.stage.toLowerCase().includes('cancelled'))) {
            
            eventSource.close();
            
            // If there was an error, throw it to be caught by the caller
            if (progress.stage.toLowerCase().includes('error')) {
              throw new Error(progress.error || 'Upload failed');
            }
          }
        } catch (error) {
          console.error('Error processing progress event:', error);
          eventSource.close();
        }
      };
      
      eventSource.onerror = () => {
        console.log('SSE connection error or closed by server');
        eventSource.close();
      };
    }
    
    return data;
  } catch (error) {
    console.error('Upload error:', error);
    throw error;
  }
}

// Cancel upload
export async function cancelUpload(): Promise<{success: boolean, message: string}> {
  try {
    // Get current session ID
    const sessionId = getUploadSessionId();
    
    if (!sessionId) {
      return {
        success: false,
        message: 'No active upload session to cancel'
      };
    }
    
    // Send cancel request to server
    const response = await apiRequest<{success: boolean, message: string}>(
      `/api/cancel-upload/${sessionId}`,
      { method: 'POST' }
    );
    
    // Clear the upload state from localStorage
    clearUploadState();
    
    // Invalidate queries to refresh UI data
    queryClient.invalidateQueries({ queryKey: ['/api/sentiment-posts'] });
    queryClient.invalidateQueries({ queryKey: ['/api/analyzed-files'] });
    queryClient.invalidateQueries({ queryKey: ['/api/disaster-events'] });
    
    return response;
  } catch (error) {
    console.error('Error cancelling upload:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Unknown error cancelling upload'
    };
  }
}

// Analyze a single text
export async function analyzeText(text: string): Promise<{
  sentiment: string;
  confidence: number;
  explanation?: string;
  disasterType?: string;
  location?: string;
}> {
  return apiRequest<{
    sentiment: string;
    confidence: number;
    explanation?: string;
    disasterType?: string;
    location?: string;
  }>('/api/analyze-text', {
    method: 'POST',
    body: JSON.stringify({ text }),
  });
}

// Delete all data
export async function deleteAllData(): Promise<{success: boolean}> {
  return apiRequest<{success: boolean}>('/api/delete-all-data', {
    method: 'DELETE',
  });
}

// Delete a sentiment post
export async function deleteSentimentPost(id: number): Promise<void> {
  return apiRequest<void>(`/api/sentiment-posts/${id}`, {
    method: 'DELETE',
  });
}

// Delete an analyzed file
export async function deleteAnalyzedFile(id: number): Promise<void> {
  return apiRequest<void>(`/api/analyzed-files/${id}`, {
    method: 'DELETE',
  });
}

// Get usage stats
export async function getUsageStats(): Promise<{
  apiLimit: number;
  apiUsed: number;
  apiRemaining: number;
  resetDate: string;
}> {
  return apiRequest<{
    apiLimit: number;
    apiUsed: number;
    apiRemaining: number;
    resetDate: string;
  }>('/api/usage-stats');
}

// Get Python console messages
export async function getPythonConsoleMessages(): Promise<{message: string, timestamp: string}[]> {
  return apiRequest<{message: string, timestamp: string}[]>('/api/python-console-messages');
}

// Export data as CSV
export async function exportDataAsCSV(): Promise<string> {
  return apiRequest<string>('/api/export-csv');
}

// Submit sentiment feedback
export async function submitSentimentFeedback(
  text: string,
  originalSentiment: string,
  correctedSentiment: string,
  postId?: number
): Promise<{success: boolean, message: string}> {
  return apiRequest<{success: boolean, message: string}>('/api/sentiment-feedback', {
    method: 'POST',
    body: JSON.stringify({
      text,
      originalSentiment,
      correctedSentiment,
      postId
    }),
  });
}

// Get sentiment feedback
export async function getSentimentFeedback(): Promise<{
  id: number;
  text: string;
  originalSentiment: string;
  correctedSentiment: string;
  postId?: number;
  trained: boolean;
  submittedAt: string;
}[]> {
  return apiRequest<{
    id: number;
    text: string;
    originalSentiment: string;
    correctedSentiment: string;
    postId?: number;
    trained: boolean;
    submittedAt: string;
  }[]>('/api/sentiment-feedback');
}

// Get untrained feedback
export async function getUntrainedFeedback(): Promise<{
  id: number;
  text: string;
  originalSentiment: string;
  correctedSentiment: string;
  postId?: number;
  trained: boolean;
  submittedAt: string;
}[]> {
  return apiRequest<{
    id: number;
    text: string;
    originalSentiment: string;
    correctedSentiment: string;
    postId?: number;
    trained: boolean;
    submittedAt: string;
  }[]>('/api/untrained-feedback');
}

// Mark feedback as trained
export async function markFeedbackAsTrained(id: number): Promise<void> {
  return apiRequest<void>(`/api/sentiment-feedback/${id}/trained`, {
    method: 'PATCH',
  });
}

// Get training examples
export async function getTrainingExamples(): Promise<{
  id: number;
  text: string;
  sentiment: string;
  createdAt: string;
}[]> {
  return apiRequest<{
    id: number;
    text: string;
    sentiment: string;
    createdAt: string;
  }[]>('/api/training-examples');
}