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
  aiTrustMessage?: string | null; // Added for validation messages in data-table
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

interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: number[][];
}

export interface AnalyzedFile {
  id: number;
  originalName: string;
  storedName: string;
  timestamp: string;
  recordCount: number;
  evaluationMetrics: EvaluationMetrics | null;
}

export interface UploadProgress {
  processed: number;
  total?: number;
  stage: string;
  error?: string;
  batchNumber?: number;
  totalBatches?: number;
  batchProgress?: number;
  currentSpeed?: number;  // Records per second
  timeRemaining?: number; // Seconds
  processingStats?: {
    successCount: number;
    errorCount: number;
    lastBatchDuration: number;
    averageSpeed: number;
  };
}

// Sentiment Posts API
export async function getSentimentPosts(filterUnknown: boolean = true): Promise<SentimentPost[]> {
  const response = await apiRequest('GET', `/api/sentiment-posts?filterUnknown=${filterUnknown}`);
  return response.json();
}

export async function getSentimentPostsByFileId(fileId: number, filterUnknown: boolean = true): Promise<SentimentPost[]> {
  const response = await apiRequest('GET', `/api/sentiment-posts/file/${fileId}?filterUnknown=${filterUnknown}`);
  return response.json();
}

// Disaster Events API
export async function getDisasterEvents(filterUnknown: boolean = true): Promise<DisasterEvent[]> {
  const response = await apiRequest('GET', `/api/disaster-events?filterUnknown=${filterUnknown}`);
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

// File Upload with simplified progress tracking

// Check if there are any active upload sessions in the database
export async function checkForActiveSessions(): Promise<string | null> {
  try {
    // Use fetch directly to get more details about the error if any
    const response = await fetch('/api/active-upload-session', {
      method: 'GET',
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      },
      credentials: 'include'
    });
    
    if (response.ok) {
      const data = await response.json();
      
      // Check if this is a stale cleared session notification
      if (data.staleSessionCleared) {
        console.log('A stale upload session was automatically cleared by the server');
        localStorage.removeItem('uploadSessionId');
        currentUploadSessionId = null;
        return null;
      }
      
      // Server reported an error but recovered
      if (data.error && data.recoverable) {
        console.warn('Server reported recoverable error:', data.errorMessage);
        // If there's no active session but we had an error, we take a conservative approach
        return data.sessionId || null;
      }
      
      if (data.sessionId) {
        // Set the current session ID
        currentUploadSessionId = data.sessionId;
        // Update localStorage for compatibility
        localStorage.setItem('uploadSessionId', data.sessionId);
        
        console.log(`Restored active upload session ${data.sessionId} from database`);
        
        // If we have progress data, use it immediately
        if (data.progress) {
          console.log('Server provided initial progress state:', data.progress);
        }
        
        return data.sessionId;
      }
      
      console.log('Active upload session check complete:', 
        data.sessionId ? `Session ${data.sessionId} active` : 'No active sessions');
      return data.sessionId || null;
    } else {
      if (response.status === 503 || response.status === 429) {
        // If the service is unavailable (503) or rate-limited (429), don't block uploads
        console.error('Server unavailable or rate-limited:', response.status);
        return null;
      }
      
      console.error('Error checking active sessions, server returned:', response.status, response.statusText);
      // Return a special value for errors that will be handled by the callers
      return 'error';
    }
  } catch (error) {
    console.error('Error checking for active upload sessions:', error);
    // Return a special value for errors that will be handled by the callers
    return 'error';
  }
}

// File Upload with enhanced progress tracking
let currentUploadController: AbortController | null = null;
let currentEventSource: EventSource | null = null;
let currentUploadSessionId: string | null = null;

// Cancel the current upload
export async function cancelUpload(): Promise<{ success: boolean; message: string }> {
  if (currentUploadSessionId) {
    try {
      // Close the event source
      if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
      }
      
      // Abort the fetch request
      if (currentUploadController) {
        currentUploadController.abort();
        currentUploadController = null;
      }
      
      // Call the server to cancel processing
      const response = await apiRequest('POST', `/api/cancel-upload/${currentUploadSessionId}`);
      const result = await response.json();
      
      // Reset the current session ID
      currentUploadSessionId = null;
      
      // Clear localStorage
      localStorage.removeItem('uploadSessionId');
      
      return result;
    } catch (error) {
      console.error('Error cancelling upload:', error);
      return { 
        success: false, 
        message: error instanceof Error ? error.message : 'Failed to cancel upload' 
      };
    }
  }
  
  return { success: false, message: 'No active upload to cancel' };
}

// Return the current upload session ID with database support
export function getCurrentUploadSessionId(): string | null {
  // First check the memory variable
  if (currentUploadSessionId) {
    return currentUploadSessionId;
  }
  
  // If not in memory, check localStorage (legacy support)
  const storedSessionId = localStorage.getItem('uploadSessionId');
  if (storedSessionId) {
    // Restore the session ID to memory
    currentUploadSessionId = storedSessionId;
    return storedSessionId;
  }
  
  return null;
}

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
  // Create a session ID for progress tracking
  const sessionId = crypto.randomUUID();
  // Store the session ID both in memory and localStorage
  currentUploadSessionId = sessionId;
  localStorage.setItem('uploadSessionId', sessionId);
  
  const formData = new FormData();
  formData.append('file', file);
  
  // Create an abort controller for the fetch request
  currentUploadController = new AbortController();
  
  // Set up EventSource for progress updates
  const eventSource = new EventSource(`/api/upload-progress/${sessionId}`);
  currentEventSource = eventSource; // Store the reference for cancel functionality
  
  // Handle progress updates
  eventSource.onmessage = (event) => {
    if (onProgress) {
      try {
        const progress = JSON.parse(event.data) as UploadProgress;
        onProgress(progress);
        
        // Check for completion or errors to clean up resources
        if (progress.stage && 
            (progress.stage.toLowerCase().includes('complete') ||
             progress.stage.toLowerCase().includes('error') ||
             progress.stage.toLowerCase().includes('cancelled'))) {
          
          // Close the event source after receiving completion status
          console.log('Upload status updated to:', progress.stage);
          setTimeout(() => {
            if (currentEventSource === eventSource) {
              currentEventSource.close();
              currentEventSource = null;
            }
          }, 1000);
        }
      } catch (error) {
        console.error('Error parsing progress data:', error);
      }
    }
  };
  
  // Handle EventSource errors
  eventSource.onerror = (error) => {
    console.error('EventSource error:', error);
    // Don't close the event source on errors as it might reconnect automatically
  };

  try {
    // Perform the upload
    const response = await fetch('/api/upload-csv', {
      method: 'POST',
      headers: {
        'X-Session-ID': sessionId
      },
      body: formData,
      credentials: 'include',
      signal: currentUploadController.signal // Add abort controller signal
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to upload CSV');
    }

    return response.json();
  } catch (error) {
    // Check if this is an abort error
    if (error.name === 'AbortError') {
      console.log('Upload was cancelled');
      throw new Error('Upload cancelled by user');
    }
    throw error;
  } finally {
    // Close the event source after a short delay
    // Don't reset the currentUploadSessionId here as it might be needed for cancellation
    setTimeout(() => {
      if (currentEventSource === eventSource) {
        currentEventSource.close();
        currentEventSource = null;
      }
    }, 2000);
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

// Delete Specific Analyzed File (CSV) and its posts
export async function deleteAnalyzedFile(id: number): Promise<{
  success: boolean;
  message: string;
}> {
  const response = await apiRequest('DELETE', `/api/analyzed-files/${id}`);
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

// Interface for Python console messages
export interface PythonConsoleMessage {
  message: string;
  timestamp: string;
}

// Get Python console messages
export async function getPythonConsoleMessages(limit: number = 100): Promise<PythonConsoleMessage[]> {
  const response = await apiRequest('GET', `/api/python-console-messages?limit=${limit}`);
  return response.json();
}

// Interface for Sentiment Feedback
export interface SentimentFeedback {
  id: number;
  originalText: string;
  originalSentiment: string;
  correctedSentiment: string;
  correctedLocation?: string | null;
  correctedDisasterType?: string | null;
  trainedOn: boolean;
  createdAt: string;
  timestamp?: string; // For backwards compatibility
  userId?: number | null;
  originalPostId?: number | null;
  possibleTrolling?: boolean;
  aiTrustMessage?: string;
  aiWarning?: string;
  updateSkipped?: boolean;
}

// Submit sentiment feedback for model improvement
export async function submitSentimentFeedback(
  originalText: string,
  originalSentiment: string,
  correctedSentiment: string,
  correctedLocation?: string,
  correctedDisasterType?: string
): Promise<SentimentFeedback & {
  status?: string;
  message?: string;
  aiTrustMessage?: string;
  performance?: {
    previous_accuracy: number;
    new_accuracy: number;
    improvement: number;
  };
}> {
  try {
    // Use fetch directly for better control of the response handling
    const response = await fetch('/api/sentiment-feedback', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        originalText,
        originalSentiment,
        correctedSentiment,
        correctedLocation,
        correctedDisasterType,
        // Don't include trainedOn as it's not in the schema and is defaulted server-side
        // Include originalPostId and userId as optional
        originalPostId: null,
        userId: null
      }),
      credentials: 'include',
    });
    
    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
    }
    
    // Get response as text first
    const textResponse = await response.text();
    console.log('Raw response from server:', textResponse);
    
    // Then try to parse as JSON
    try {
      const jsonResponse = JSON.parse(textResponse);
      console.log('Successfully parsed sentiment feedback response:', jsonResponse);
      return jsonResponse;
    } catch (parseError) {
      console.error("Failed to parse JSON response:", parseError, "Raw text:", textResponse);
      throw new Error("Invalid JSON in response");
    }
  } catch (error) {
    console.error("Error submitting sentiment feedback:", error);
    // Return a basic object if the request or JSON parse fails
    return {
      id: 0,
      originalText,
      originalSentiment,
      correctedSentiment,
      correctedLocation,
      correctedDisasterType,
      trainedOn: false,
      createdAt: new Date().toISOString(),
      timestamp: new Date().toISOString(), // Include for backwards compatibility
      status: "error",
      message: "Failed to submit feedback",
      originalPostId: null,
      userId: null,
      possibleTrolling: false,
      aiTrustMessage: "Error communicating with server"
    };
  }
}

// Get all sentiment feedback
export async function getSentimentFeedback(): Promise<SentimentFeedback[]> {
  const response = await apiRequest('GET', '/api/sentiment-feedback');
  return response.json();
}

// Get untrained feedback for model retraining
export async function getUntrainedFeedback(): Promise<SentimentFeedback[]> {
  const response = await apiRequest('GET', '/api/untrained-feedback');
  return response.json();
}

// Mark sentiment feedback as trained
export async function markFeedbackAsTrained(id: number): Promise<{message: string}> {
  const response = await apiRequest('PATCH', `/api/sentiment-feedback/${id}/trained`);
  return response.json();
}