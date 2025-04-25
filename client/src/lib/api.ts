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
  labels: string[];
  trainingDate: string;
  testSize: number;
  sampleCount: number;
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
  autoCloseDelay?: number; // Time in ms to auto-close "Analysis complete" state
  timestamp?: number;      // Timestamp for ordering updates
  processingStats?: {
    successCount: number;
    errorCount: number;
    lastBatchDuration: number;
    averageSpeed: number;
  };
}

// Sentiment Posts API
export async function getSentimentPosts(filterUnknown: boolean = false): Promise<SentimentPost[]> {
  const response = await apiRequest('GET', `/api/sentiment-posts?filterUnknown=${filterUnknown}`);
  return response.json();
}

export async function getSentimentPostsByFileId(fileId: number, filterUnknown: boolean = false): Promise<SentimentPost[]> {
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

// Standard analysis mode is the only option (hybrid model removed)
let currentModelType: 'standard' = 'standard';

// Cancel the current upload with optional gentle/force modes
export async function cancelUpload(forceCancel = false): Promise<{ success: boolean; message: string; forceCloseCalled?: boolean }> {
  const sessionId = currentUploadSessionId || localStorage.getItem('uploadSessionId');
  
  if (!sessionId) {
    console.log('No session ID found - nothing to cancel');
    return { success: false, message: 'No active upload to cancel' };
  }
  
  // IMPROVED: Add gentler cancellation by default
  if (forceCancel) {
    console.log('🔥 FORCE CANCEL MODE ACTIVATED');
    
    // Force cancel immediately cleans up client side regardless of server response
    
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
    
    // Reset variables
    currentUploadSessionId = null;
    
    // Clean up localStorage
    localStorage.removeItem('uploadSessionId');
    localStorage.removeItem('isUploading');
    localStorage.removeItem('uploadProgress');
    localStorage.removeItem('lastProgressTimestamp');
    localStorage.removeItem('lastUIUpdateTimestamp');
    localStorage.removeItem('serverRestartProtection');
    localStorage.removeItem('serverRestartTimestamp');
    
    // Try to notify other tabs about the force cancel
    try {
      if (window.BroadcastChannel) {
        const bc = new BroadcastChannel('upload_status');
        bc.postMessage({
          type: 'upload_force_cancelled',
          timestamp: Date.now()
        });
        bc.close();
      }
    } catch (e) {
      console.error('Error broadcasting force cancel:', e);
    }
  } else {
    // NEW GENTLE APPROACH: For gentle cancel, log but don't immediately close everything
    console.log('🌸 GENTLE CANCEL MODE - Letting server handle the cancellation gracefully');
    
    // Don't immediately terminate connections - let the server complete current batch
    // This provides a smoother experience and animates the "Upload Canceled" state
  }
  
  if (sessionId) {
    try {
      // For gentle cancel, we DON'T force close connections immediately
      // Let the server finish any in-progress work and animate the cancellation
      if (forceCancel) {
        // Only force close these if in force cancel mode 
        // (already done earlier in the force block, but double check here)
        if (currentEventSource) {
          currentEventSource.close();
          currentEventSource = null;
        }
        
        if (currentUploadController) {
          currentUploadController.abort();
          currentUploadController = null;
        }
      }
      
      // Always call the server to cancel - the server knows how to handle both force and gentle cancels
      const response = await apiRequest('POST', `/api/cancel-upload/${sessionId}`);
      const result = await response.json();
      
      // For gentle cancel, we leave session cleanup for after the success message
      // This allows the UI to show the "Upload Canceled" message and animate first
      
      // In gentle mode, we actually want to keep things active until we confirm server completed
      if (forceCancel) {
        // Reset the current session ID immediately in force mode
        currentUploadSessionId = null;
      } else {
        // For gentle mode, we delay cleanup to allow the animation
        setTimeout(() => {
          // Now we can clean up this reference
          currentUploadSessionId = null;
        }, 1500);
      }
      
      // Clear localStorage (but only if not force cancel mode, which already did this)
      if (!forceCancel) {
        // Don't immediately remove these in gentle mode to allow UI transitions
        setTimeout(() => {
          localStorage.removeItem('uploadSessionId');
        }, 1500);
      }
      
      // If server cancellation failed but we're in force mode, still return success
      if (!result.success && forceCancel) {
        return { 
          success: true, 
          message: 'Force canceled. Client state cleared but server failed to cancel.',
          forceCloseCalled: true
        };
      }
      
      return {
        ...result,
        forceCloseCalled: forceCancel
      };
    } catch (error) {
      console.error('Error cancelling upload:', error);
      
      // If force cancel is true, still return success
      if (forceCancel) {
        return { 
          success: true, 
          message: 'Force canceled. Client state cleared but error occurred with server.',
          forceCloseCalled: true
        };
      }
      
      return { 
        success: false, 
        message: error instanceof Error ? error.message : 'Failed to cancel upload' 
      };
    }
  }
  
  // If we get here with force cancel, still return success
  if (forceCancel) {
    return { 
      success: true, 
      message: 'Force canceled. No active upload found but client state cleared.',
      forceCloseCalled: true
    };
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
    
    // ENHANCED: Send the session ID from localStorage as a query parameter
    // This allows the server to check specifically for this session
    const endpoint = cachedSessionId 
      ? `/api/active-upload-session?sessionId=${encodeURIComponent(cachedSessionId)}`
      : '/api/active-upload-session';
      
    const response = await apiRequest('GET', endpoint);
    
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
  errorRecovered?: boolean; // Add flag for error recovery
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

  // Keep track of the last timestamp we processed to avoid out-of-order updates
  let lastProcessedTimestamp = 0;
  
  // Store recent progress updates keyed by timestamp for deduplication
  const recentProgressUpdates = new Map<number, boolean>();
  
  eventSource.onmessage = (event) => {
    try {
      const progress = JSON.parse(event.data) as UploadProgress;
      
      // Add timestamp if missing
      if (!progress.timestamp) {
        progress.timestamp = Date.now();
      }
      
      // Anti-flicker: Check if we've seen this exact update before based on timestamp
      if (recentProgressUpdates.has(progress.timestamp)) {
        console.log(`🎭 Suppressing duplicate progress update from timestamp ${progress.timestamp}`);
        return;
      }
      
      // Log the reception without details to reduce console noise
      console.log('Progress event received:', progress);
      
      // Record this timestamp to prevent duplicate processing
      recentProgressUpdates.set(progress.timestamp, true);
      
      // Clean up old entries to avoid memory leaks
      if (recentProgressUpdates.size > 20) {
        // Keep only the 10 most recent entries
        const keys = Array.from(recentProgressUpdates.keys()).sort((a, b) => a - b);
        for (let i = 0; i < keys.length - 10; i++) {
          recentProgressUpdates.delete(keys[i]);
        }
      }
      
      // Anti-flicker: Ensure events are processed in chronological order
      // Only process events that are newer than the last one we processed
      if (progress.timestamp < lastProcessedTimestamp - 1000) {
        console.log(`🎭 Ignoring out-of-order progress update: ${progress.timestamp} < ${lastProcessedTimestamp}`);
        return;
      }
      
      // Special case: Always process terminal state messages (errors, completion) 
      // BUT STRICT FILTER: Only exact "Analysis complete" is treated as completion
      // NOT "Completed record X/Y" which is NOT a terminal state!
      const stageLower = progress.stage?.toLowerCase() || '';
      const isTerminalState = progress.stage === 'Analysis complete' || 
                             stageLower === 'analysis complete' || 
                             (stageLower.includes('complete') && !stageLower.includes('record')) ||
                             progress.stage === 'Upload Error' || 
                             progress.error;
      
      if (isTerminalState) {
        console.log(`🚨 TERMINAL STATE DETECTED! AUTO-CLOSING!`, progress.stage?.toLowerCase());
        
        // For completion, force progress to 100%
        if (progress.stage?.toLowerCase()?.includes('complete') && progress.total && progress.total > 0) {
          progress.processed = progress.total || 10; // Default to 10 if total is undefined
          progress.stage = 'Analysis complete';
        }
        
        // Close the EventSource connection - we're done!
        console.log('Closing EventSource connection');
        eventSource.close();
        currentEventSource = null;
        
        // Always process terminal states immediately
        if (onProgress) {
          console.log('Success completion detected - refreshing data');
          onProgress(progress);
        }
        
        return;
      }
      
      // Update last processed timestamp
      if (progress.timestamp > lastProcessedTimestamp) {
        lastProcessedTimestamp = progress.timestamp;
      }
      
      // For normal updates, pass to callback with timestamps for the UI to handle debouncing
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
  // Clean up any upload-related localStorage items before text analysis
  // This prevents the upload modal from appearing when analyzing text
  if (localStorage.getItem('isUploading') || localStorage.getItem('uploadSessionId')) {
    console.log('🧹 Cleaning up stale upload state before text analysis');
    
    // Clear upload-related localStorage items
    localStorage.removeItem('isUploading');
    localStorage.removeItem('uploadProgress');
    localStorage.removeItem('uploadSessionId');
    localStorage.removeItem('lastProgressTimestamp');
    localStorage.removeItem('lastUIUpdateTimestamp');
    localStorage.removeItem('serverRestartProtection');
    localStorage.removeItem('serverRestartTimestamp');
    
    // Also try to clean up any error sessions on the server
    try {
      cleanupErrorSessions().catch(e => console.error('Error cleaning up sessions:', e));
    } catch (e) {
      console.error('Error initiating session cleanup:', e);
    }
  }
  
  // Make the API request with clean state
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

// Text processing interface
export interface TextProcessingResult {
  normalizedText: string;
  tokenizedText: string[];
  stemmedText: string[];
  finalOutput: string;
}

// Process text through NLP pipeline
export async function processText(text: string): Promise<TextProcessingResult> {
  const response = await apiRequest('POST', '/api/text-processing', { text });
  return response.json();
}

// Hybrid model functionality has been removed

// Real machine learning model training functions

/**
 * Use a custom demo file for sentiment analysis
 */
export async function useCustomDemoFile(): Promise<{ 
  fileId: number;
  metrics: EvaluationMetrics; 
}> {
  // Start a progress tracking event source
  const sessionId = crypto.randomUUID();
  currentUploadSessionId = sessionId;
  localStorage.setItem('uploadSessionId', sessionId);
  
  console.log('[SENTIMENT-ANALYSIS] Starting sentiment analysis with session ID:', sessionId);
  
  // Track the progress through SSE
  const eventSource = new EventSource(`/api/upload-progress/${sessionId}`);
  currentEventSource = eventSource;
  
  // Set up event handlers outside try block to ensure they're registered
  let progressPromise = new Promise<{ fileId: number; metrics: EvaluationMetrics }>((resolve, reject) => {
    const timeout = setTimeout(() => {
      console.error('[SENTIMENT-ANALYSIS] Processing timed out after 5 minutes');
      eventSource.close();
      reject(new Error('Sentiment analysis timed out after 5 minutes'));
    }, 300000); // 5 minutes timeout
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('[SENTIMENT-ANALYSIS] Progress update:', data);
        
        // Check if processing is complete
        if (data.type === 'complete' || 
            (data.progress && data.progress.processed === data.progress.total)) {
          console.log('[SENTIMENT-ANALYSIS] Processing complete:', data);
          clearTimeout(timeout);
          eventSource.close();
          
          if (data.error) {
            console.error('[SENTIMENT-ANALYSIS] Error in completion event:', data.error);
            reject(new Error(data.error));
          } else {
            resolve({
              fileId: data.fileId,
              metrics: data.metrics
            });
          }
        }
      } catch (error) {
        console.error('[SENTIMENT-ANALYSIS] Error parsing event data:', error);
      }
    };
    
    eventSource.onerror = (err) => {
      console.error('[SENTIMENT-ANALYSIS] EventSource error:', err);
      clearTimeout(timeout);
      eventSource.close();
      reject(new Error('Error with event source connection'));
    };
  });
  
  try {
    // Call the server endpoint
    console.log('[SENTIMENT-ANALYSIS] Calling server endpoint');
    const response = await apiRequest('POST', '/api/use-custom-demo-file');
    
    if (!response.ok) {
      // Try to parse error information from the response
      let errorMessage = 'Failed to initialize sentiment analysis';
      try {
        const errorData = await response.json();
        console.error('[SENTIMENT-ANALYSIS] Server error response:', errorData);
        errorMessage = errorData.details || errorData.error || errorMessage;
      } catch (e) {
        console.error('[SENTIMENT-ANALYSIS] Could not parse error response', e);
      }
      throw new Error(errorMessage);
    }
    
    const data = await response.json();
    console.log('[SENTIMENT-ANALYSIS] Server initialization response:', data);
    
    // Wait for the completion of the progress promise
    return await progressPromise;
    
  } catch (error) {
    console.error('[SENTIMENT-ANALYSIS] Error:', error);
    
    // Clean up the event source
    if (eventSource) {
      eventSource.close();
    }
    
    throw error;
  }
}

/**
 * Create a demo dataset and train a model with it
 */
export async function createDemoDataset(
  recordCount: number = 100,
  onProgress?: (progress: UploadProgress) => void
): Promise<{
  success: boolean;
  fileId: number;
  metrics: EvaluationMetrics;
  message: string;
  sessionId: string;
}> {
  const controller = new AbortController();
  currentUploadController = controller;
  const { signal } = controller;
  
  try {
    // Make the request to create a demo dataset
    const response = await apiRequest('POST', '/api/create-demo-dataset', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ recordCount }),
      signal,
    });
    
    if (!response.ok) {
      throw new Error('Failed to create demo dataset');
    }
    
    const result = await response.json();
    
    // If onProgress callback is provided and session ID is returned
    if (onProgress && result.sessionId) {
      // Set up event source for progress updates
      const eventSource = new EventSource(`/api/upload-progress/${result.sessionId}`);
      currentEventSource = eventSource;
      
      // Track last processed timestamp to avoid out-of-order updates
      let lastProcessedTimestamp = 0;
      
      eventSource.onmessage = (event) => {
        try {
          const progress = JSON.parse(event.data);
          
          // Skip stale updates
          if (progress.timestamp && progress.timestamp <= lastProcessedTimestamp) {
            return;
          }
          
          // Update last processed timestamp
          if (progress.timestamp) {
            lastProcessedTimestamp = progress.timestamp;
          }
          
          // Call the progress callback
          onProgress(progress);
          
          // Close the event source when training is complete
          if (progress.stage?.toLowerCase()?.includes('complete')) {
            setTimeout(() => {
              eventSource.close();
              currentEventSource = null;
            }, 1500);
          }
        } catch (error) {
          console.error('Error parsing progress update:', error);
        }
      };
      
      // Handle event source errors
      eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        eventSource.close();
        currentEventSource = null;
      };
    }
    
    return result;
  } catch (error) {
    console.error('Error creating demo dataset:', error);
    throw error;
  }
}

/**
 * Train a model with an existing file
 */
export async function trainModel(
  fileId: number,
  onProgress?: (progress: UploadProgress) => void
): Promise<{
  success: boolean;
  metrics: EvaluationMetrics;
  message: string;
  sessionId: string;
}> {
  const controller = new AbortController();
  currentUploadController = controller;
  const { signal } = controller;
  
  try {
    // Make the request to train the model
    const response = await apiRequest('POST', `/api/train-model/${fileId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      signal,
    });
    
    if (!response.ok) {
      throw new Error('Failed to train model');
    }
    
    const result = await response.json();
    
    // If onProgress callback is provided and session ID is returned
    if (onProgress && result.sessionId) {
      // Set up event source for progress updates
      const eventSource = new EventSource(`/api/upload-progress/${result.sessionId}`);
      currentEventSource = eventSource;
      
      // Track last processed timestamp to avoid out-of-order updates
      let lastProcessedTimestamp = 0;
      
      eventSource.onmessage = (event) => {
        try {
          const progress = JSON.parse(event.data);
          
          // Skip stale updates
          if (progress.timestamp && progress.timestamp <= lastProcessedTimestamp) {
            return;
          }
          
          // Update last processed timestamp
          if (progress.timestamp) {
            lastProcessedTimestamp = progress.timestamp;
          }
          
          // Call the progress callback
          onProgress(progress);
          
          // Close the event source when training is complete
          if (progress.stage?.toLowerCase()?.includes('complete')) {
            setTimeout(() => {
              eventSource.close();
              currentEventSource = null;
            }, 1500);
          }
        } catch (error) {
          console.error('Error parsing progress update:', error);
        }
      };
      
      // Handle event source errors
      eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        eventSource.close();
        currentEventSource = null;
      };
    }
    
    return result;
  } catch (error) {
    console.error('Error training model:', error);
    throw error;
  }
}