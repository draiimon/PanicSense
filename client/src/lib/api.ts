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

// Check if there are any active upload sessions in the database
// DATABASE BOSS! DATABASE IS THE SOURCE OF TRUTH! LOCALSTORAGE IS JUST THE ASSISTANT
export async function checkForActiveSessions(): Promise<string | null> {
  try {
    // Quick localStorage check for fast UI while database check happens
    const cachedSessionId = localStorage.getItem('uploadSessionId');
    const isUploadingCache = localStorage.getItem('isUploading') === 'true';
    
    // Check if we've recently asked the database to avoid too many calls
    const cacheKey = 'lastDatabaseCheck';
    const lastCheckTime = parseInt(localStorage.getItem(cacheKey) || '0');
    const now = Date.now();
    const minCheckInterval = 5000; // 5 seconds minimum between full checks
    
    // For immediate response while rate limiting database calls
    if (now - lastCheckTime < minCheckInterval && cachedSessionId && isUploadingCache) {
      // Schedule background verification with database boss
      setTimeout(() => {
        checkForActiveSessions();
      }, 3000);
      
      return cachedSessionId; // Return cached for immediate UI display
    }
    
    // Mark that we're doing a database check
    localStorage.setItem(cacheKey, now.toString());
    
    // ALWAYS ask the database (boss) for the truth!
    console.log('📊 Asking database boss for active sessions');
    const response = await apiRequest('GET', '/api/active-upload-session');
    
    if (!response.ok) {
      throw new Error('Database check failed');
    }
    
    const data = await response.json();
    
    // Handle server restart detection
    if (data.serverRestartDetected) {
      console.log('⚠️ Server restart detected! Must follow database rules');
    }
    
    // === HANDLE DATABASE RESPONSE ===
    if (data.sessionId) {
      // === BOSS SAYS YES: ACTIVE SESSION EXISTS ===
      console.log('👑 DATABASE BOSS CONFIRMS: Active session ' + data.sessionId);
      
      // Update everything according to database (the boss)
      currentUploadSessionId = data.sessionId;
      localStorage.setItem('uploadSessionId', data.sessionId);
      localStorage.setItem('isUploading', 'true');
      
      // Handle progress data if available
      if (data.progress) {
        try {
          // Parse progress if it's a string
          let bossProgress = typeof data.progress === 'string' 
            ? JSON.parse(data.progress)
            : data.progress;
            
          // Add timestamps and mark as official database data
          const officialData = {
            ...bossProgress,
            timestamp: Date.now(),
            savedAt: Date.now(),
            bossData: true // Flag from database
          };
          
          // Save to localStorage for fast access
          localStorage.setItem('uploadProgress', JSON.stringify(officialData));
        } catch (e) {
          console.error('Error handling database progress data:', e);
        }
      }
      
      return data.sessionId;
    } else {
      // === BOSS SAYS NO: NO ACTIVE SESSION ===
      console.log('👑 DATABASE BOSS SAYS: No active sessions exist');
      
      if (data.staleSessionCleared) {
        console.log('🧹 Boss cleaned stale session on server');
      }
      
      // Clear localStorage to match database state
      localStorage.removeItem('isUploading');
      localStorage.removeItem('uploadProgress');
      localStorage.removeItem('uploadSessionId');
      localStorage.removeItem('lastProgressTimestamp');
      localStorage.removeItem('lastUIUpdateTimestamp');
      
      // Check if localStorage needs to retain session for UI stability
      const localSession = localStorage.getItem('uploadSessionId');
      if (localSession && localStorage.getItem('uploadProgress')) {
        try {
          const progress = JSON.parse(localStorage.getItem('uploadProgress') || '{}');
          const savedAt = progress.savedAt || 0;
          const fiveMinutesAgo = Date.now() - (5 * 60 * 1000);
          
          // Only keep very recent sessions to prevent stale UI
          if (savedAt >= fiveMinutesAgo) {
            console.log('Recent localStorage session kept for UI stability:', localSession);
            return localSession;
          }
        } catch (e) {
          // Ignore parse errors
        }
      }
      
      console.log('Active upload session check complete: No active sessions');
      return null;
    }
  } catch (error) {
    console.error('Error checking for active sessions:', error);
    
    // On error, fall back to localStorage for UI stability
    const localSessionId = localStorage.getItem('uploadSessionId');
    if (localSessionId && localStorage.getItem('isUploading') === 'true') {
      return localSessionId;
    }
    
    return null;
  }
}

// File Upload with enhanced progress tracking and cancellation
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
  // Clean up any existing uploads first
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
  
  if (currentUploadController) {
    currentUploadController.abort();
    currentUploadController = null;
  }
  
  // Create a new abort controller
  currentUploadController = new AbortController();
  const { signal } = currentUploadController;
  
  const formData = new FormData();
  formData.append('file', file);

  // Generate a unique session ID
  const sessionId = crypto.randomUUID();
  currentUploadSessionId = sessionId;
  
  // Store the session ID in both localStorage (legacy) and database
  localStorage.setItem('uploadSessionId', sessionId);
  
  // If onProgress callback is provided, check for any active upload sessions 
  // that might have been interrupted
  if (onProgress) {
    try {
      // Attempt to restore any active session from previous runs
      const activeSessionId = await checkForActiveSessions();
      if (activeSessionId) {
        console.log(`Detected active upload session: ${activeSessionId}. Using existing session.`);
        // If there's an active session in the database, use that instead
        currentUploadSessionId = activeSessionId;
      }
    } catch (error) {
      console.error('Error checking for active sessions:', error);
    }
  }

  // Set up event source for progress updates using the potentially updated sessionId
  const eventSource = new EventSource(`/api/upload-progress/${currentUploadSessionId}`);
  currentEventSource = eventSource;

  eventSource.onmessage = (event) => {
    try {
      const progress = JSON.parse(event.data) as UploadProgress;
      console.log('Progress event received:', progress);

      if (onProgress) {
        console.log('Progress being sent to UI:', progress);
        onProgress(progress);
      }
    } catch (error) {
      console.error('Error parsing progress data:', error);
    }
  };

  try {
    const response = await fetch('/api/upload-csv', {
      method: 'POST',
      headers: {
        'X-Session-ID': currentUploadSessionId  // Use the potentially updated sessionId
      },
      body: formData,
      credentials: 'include',
      signal, // Add abort signal
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to upload CSV');
    }

    return response.json();
  } catch (error: any) {
    if (error?.name === 'AbortError') {
      throw new Error('Upload was cancelled');
    }
    throw error;
  } finally {
    eventSource.close();
    currentEventSource = null;
    currentUploadSessionId = null;
    // Clear the session ID from localStorage
    localStorage.removeItem('uploadSessionId');
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

// Reset all upload sessions (admin endpoint)
export async function resetUploadSessions(): Promise<{
  success: boolean;
  message: string;
}> {
  const response = await fetch('/api/reset-upload-sessions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ all: true })
  });
  
  // Clean up localStorage
  localStorage.removeItem('isUploading');
  localStorage.removeItem('uploadProgress');
  localStorage.removeItem('uploadSessionId');
  localStorage.removeItem('lastProgressTimestamp');
  
  return response.json();
}

// Clean up any error or stale sessions in the database
export async function cleanupErrorSessions(): Promise<{
  success: boolean;
  clearedCount: number;
  message: string;
}> {
  const response = await fetch('/api/cleanup-error-sessions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    }
  });
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