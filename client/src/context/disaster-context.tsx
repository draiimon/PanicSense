import React, { createContext, ReactNode, useContext, useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { useToast } from "@/hooks/use-toast";
import { 
  getSentimentPosts, 
  getDisasterEvents, 
  getAnalyzedFiles,
  SentimentPost,
  DisasterEvent,
  AnalyzedFile,
  getCurrentUploadSessionId,
  checkForActiveSessions
} from "@/lib/api";

// Declare the window extension for EventSource tracking
declare global {
  interface Window {
    _activeEventSources?: {
      [key: string]: EventSource;
    };
  }
}

// Type definitions
interface ProcessingStats {
  successCount: number;
  errorCount: number;
  averageSpeed: number;
}

interface UploadProgress {
  processed: number;
  total: number;
  stage: string;
  timestamp?: number;  // Add timestamp to ensure proper ordering of updates
  batchNumber?: number;
  totalBatches?: number;
  batchProgress?: number;
  currentSpeed?: number;
  timeRemaining?: number;
  processingStats?: ProcessingStats;
  error?: string;
}

interface DisasterContextType {
  // Data
  sentimentPosts: SentimentPost[];
  disasterEvents: DisasterEvent[];
  analyzedFiles: AnalyzedFile[];

  // Loading states
  isLoadingSentimentPosts: boolean;
  isLoadingDisasterEvents: boolean;
  isLoadingAnalyzedFiles: boolean;
  isUploading: boolean;

  // Upload progress
  uploadProgress: UploadProgress;

  // Error states
  errorSentimentPosts: Error | null;
  errorDisasterEvents: Error | null;
  errorAnalyzedFiles: Error | null;

  // Stats
  activeDiastersCount: number;
  analyzedPostsCount: number;
  dominantSentiment: string;
  dominantDisaster: string;
  modelConfidence: number;
  
  // Sentiment statistics
  dominantSentimentPercentage?: number;
  secondDominantSentiment?: string | null;
  secondDominantSentimentPercentage?: number;
  sentimentPercentages?: Record<string, number>;
  
  // Disaster statistics
  dominantDisasterPercentage?: number;
  secondDominantDisaster?: string | null;
  secondDominantDisasterPercentage?: number;
  disasterPercentages?: Record<string, number>;

  // Filters
  selectedDisasterType: string;
  setSelectedDisasterType: (type: string) => void;

  // Upload state management
  setIsUploading: (state: boolean) => void;
  setUploadProgress: (progress: UploadProgress) => void;

  // Refresh function
  refreshData: () => void;
}

const DisasterContext = createContext<DisasterContextType | undefined>(undefined);

// Initial states
const initialProgress: UploadProgress = {
  processed: 0,
  total: 0,
  stage: "Initializing...",
  timestamp: 0,
  batchNumber: 0,
  totalBatches: 0,
  batchProgress: 0,
  currentSpeed: 0,
  timeRemaining: 0
};

export function DisasterContextProvider({ children }: { children: ReactNode }): JSX.Element {
  // Create a ref to track if initial session check has been performed
  const sessionCheckPerformedRef = useRef(false);
  
  // Check localStorage for existing upload state, but don't trust it completely
  // We'll validate with the database check
  const storedIsUploading = localStorage.getItem('isUploading') === 'true';
  const storedProgress = localStorage.getItem('uploadProgress');
  const storedSessionId = localStorage.getItem('uploadSessionId');
  
  // Initialize with localStorage data, but we'll override based on database
  let initialUploadState = storedIsUploading;
  let initialProgressState = initialProgress;
  
  // If we have stored progress, parse it
  if (storedProgress) {
    try {
      const parsedProgress = JSON.parse(storedProgress);
      initialProgressState = parsedProgress;
    } catch (error) {
      console.error('Failed to parse stored upload progress', error);
    }
  }
  
  // State 
  const [selectedDisasterType, setSelectedDisasterType] = useState<string>("All");
  const [isUploading, setIsUploading] = useState<boolean>(initialUploadState);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>(initialProgressState);

  // Get toast for notifications
  const { toast } = useToast();
  
  // Persistence for upload state
  // Store upload state in localStorage when it changes
  useEffect(() => {
    if (isUploading) {
      try {
        // Store with timestamp to ensure we have the most recent data
        const dataToStore = {
          ...uploadProgress,
          savedAt: Date.now() // Add timestamp for freshness check
        };
        
        // Store the current upload progress and state
        localStorage.setItem('isUploading', 'true');
        localStorage.setItem('uploadProgress', JSON.stringify(dataToStore));
        
        // Log persistence for debugging
        console.log('Saved upload progress to localStorage:', dataToStore);
      } catch (error) {
        // Handle storage errors gracefully
        console.error('Failed to store upload progress:', error);
      }
    } else {
      // Clear storage when upload is finished
      localStorage.removeItem('isUploading');
      localStorage.removeItem('uploadProgress');
    }
  }, [isUploading, uploadProgress]);

  // Get current location to detect route changes
  const [location] = useLocation();

  // Prepare refresh function to use with data loading
  const refreshDataRef = useRef<() => void>(() => {});
  
  // Queries for data loading
  const { 
    data: sentimentPosts = [], 
    isLoading: isLoadingSentimentPosts,
    error: errorSentimentPosts,
    refetch: refetchSentimentPosts
  } = useQuery({ 
    queryKey: ['/api/sentiment-posts'],
    queryFn: () => getSentimentPosts(),
    staleTime: 10 * 60 * 1000, // 10 minutes
    retry: 3 // Retry 3 times if failed
  });

  const { 
    data: disasterEvents = [], 
    isLoading: isLoadingDisasterEvents,
    error: errorDisasterEvents,
    refetch: refetchDisasterEvents
  } = useQuery({ 
    queryKey: ['/api/disaster-events'],
    queryFn: () => getDisasterEvents(),
    staleTime: 10 * 60 * 1000, // 10 minutes
    retry: 3 // Retry 3 times if failed
  });

  const { 
    data: analyzedFiles = [], 
    isLoading: isLoadingAnalyzedFiles,
    error: errorAnalyzedFiles,
    refetch: refetchAnalyzedFiles
  } = useQuery({ 
    queryKey: ['/api/analyzed-files'],
    queryFn: () => getAnalyzedFiles(),
    staleTime: 10 * 60 * 1000, // 10 minutes
    retry: 3 // Retry 3 times if failed
  });
  
  // Update refresh function after queries are initialized
  useEffect(() => {
    // Define refresh function using the refetch functions from queries
    refreshDataRef.current = () => {
      if (refetchSentimentPosts && refetchDisasterEvents && refetchAnalyzedFiles) {
        refetchSentimentPosts();
        refetchDisasterEvents();
        refetchAnalyzedFiles();
      }
    };
  }, [refetchSentimentPosts, refetchDisasterEvents, refetchAnalyzedFiles]);
  
  // Helper function to access the refresh function
  const refreshData = () => refreshDataRef.current();
  
  // The check and reconnect function
  // Improved version with anti-flickering safeguards
  const checkAndReconnectToActiveUploads = async () => {
    // ANTI-FLICKERING: Check localStorage first to maintain UI stability
    const storedSessionId = localStorage.getItem('uploadSessionId');
    const storedIsUploading = localStorage.getItem('isUploading') === 'true';
    let storedProgress = null;
    
    // ******************************************
    // THIS IS THE KEY ANTI-FLICKERING TECHNIQUE:
    // IMMEDIATELY APPLY THE LOCALSTORAGE STATE
    // ******************************************    
    // If localStorage shows as uploading, immediately restore that state
    // This creates a smooth non-flickering experience by never "turning off" the modal
    // until we're 100% sure the upload is done
    if (storedIsUploading && storedSessionId && !isUploading) {
      console.log("📱 Immediately restoring upload state from localStorage:", storedSessionId);
      // Immediately set UI state to uploading to prevent flickering
      setIsUploading(true);
      
      try {
        const progressData = localStorage.getItem('uploadProgress');
        if (progressData) {
          storedProgress = JSON.parse(progressData);
          // Prevent showing an empty progress state
          if (storedProgress && storedProgress.processed > 0) {
            setUploadProgress(storedProgress);
          }
        }
      } catch (e) {
        // Silently handle parse errors
      }
    }
    
    try {
      // THEN check database for active uploads - but UI already shows upload in progress
      // This way, even if the API is slow, the user still sees consistent state
      const activeSessionId = await checkForActiveSessions();
      
      // If found an active session in database, this is the authority source of truth
      if (activeSessionId) {
        // No need to log every time - reduces console spam
        console.log('Active upload session confirmed by database:', activeSessionId);
        
        // Database tells us we're active, so resurrect localStorage state too
        setIsUploading(true);
        localStorage.setItem('isUploading', 'true');
        localStorage.setItem('uploadSessionId', activeSessionId);
        
        // Create a minimal progress object if it doesn't exist yet
        // This ensures we always have something to display
        const storedProgress = localStorage.getItem('uploadProgress');
        if (!storedProgress) {
          console.log('Database session active but no progress in localStorage, fetching data');
          try {
            const response = await fetch(`/api/upload-progress/${activeSessionId}`);
            if (response.ok) {
              const progressEvent = await response.json();
              if (progressEvent) {
                // IMPORTANT: Add timestamp to ensure proper ordering
                progressEvent.timestamp = progressEvent.timestamp || Date.now();
                setUploadProgress(progressEvent);
                localStorage.setItem('uploadProgress', JSON.stringify({
                  ...progressEvent,
                  savedAt: Date.now()
                }));
              }
            }
          } catch (err) {
            // Silently handle errors to prevent console spam
          }
        }
        
        // Set up a more robust EventSource for progress updates with deduplication
        // but ONLY if one doesn't already exist to prevent duplicates
        const handleMessage = (event: MessageEvent) => {
          try {
            if (!event.data) return;
            
            // Parse progress data
            const progress = JSON.parse(event.data);
            
            // Log the progress data
            console.log("Progress event received:", progress);
            
            // Important: Check timestamps - only update if this is newer than our last update
            // This prevents out-of-order updates from causing flickering
            const currentTimestamp = progress.timestamp || Date.now();
            const lastTimestamp = parseInt(localStorage.getItem('lastProgressTimestamp') || '0');
            
            // Anti-flicker protection
            const lastUIUpdate = parseInt(localStorage.getItem('lastUIUpdateTimestamp') || '0');
            const now = Date.now();
            const timeSinceLastUpdate = now - lastUIUpdate;
            
            // Only update if this is a newer message
            if (currentTimestamp >= lastTimestamp) {
              // Store the latest progress in localStorage immediately
              // This ensures we don't lose data even if UI updates are debounced
              localStorage.setItem('uploadProgress', JSON.stringify({
                ...progress,
                savedAt: now
              }));
              localStorage.setItem('lastProgressTimestamp', currentTimestamp.toString());
              
              // DEBOUNCE UI updates - limit to one update per 300ms to prevent flickering
              // BUT make exceptions for important state changes that should be immediate
              const isImportantStateChange = 
                // Always show immediately when batch changes
                progress.batchNumber !== uploadProgress.batchNumber ||
                // Always show immediately when state changes (complete, error, etc)
                progress.stage !== uploadProgress.stage ||
                // Always update if we haven't updated in over 500ms
                timeSinceLastUpdate > 500;
                
              if (isImportantStateChange || timeSinceLastUpdate > 300) {
                // Log what we're sending to the UI
                console.log("Progress being sent to UI:", progress);
                
                // Update UI with progress
                setUploadProgress(progress);
                
                // Record the time of this UI update
                localStorage.setItem('lastUIUpdateTimestamp', now.toString());
              }
            }
            
            // If the upload is complete or has an error, close the connection
            const stageLower = progress.stage.toLowerCase();
            if (stageLower.includes('complete') || 
                stageLower.includes('error') ||
                stageLower.includes('cancelled')) {
              
              // Check if eventSource still exists before closing
              if (window._activeEventSources?.[activeSessionId]) {
                window._activeEventSources[activeSessionId].close();
                if (window._activeEventSources) {
                  delete window._activeEventSources[activeSessionId];
                }
              }
              
              // If it completed successfully, refresh data to show new records
              if (progress.stage.toLowerCase().includes('complete')) {
                // First refresh data, but wait a bit for server to finalize everything
                setTimeout(() => {
                  refreshData();
                }, 500);
              }
              
              // Use a timer before closing the modal to prevent abrupt UI changes
              // This gives user time to see the final status
              const completionDelay = stageLower.includes('error') ? 2000 : 3000;
              
              // For completion states, we'll delay the modal close slightly
              // This prevents abrupt UI changes and makes the experience smoother
              setTimeout(() => {
                // Clear the upload progress from localStorage immediately
                localStorage.removeItem('isUploading');
                localStorage.removeItem('uploadProgress');
                localStorage.removeItem('uploadSessionId');
                localStorage.removeItem('lastProgressTimestamp');
                localStorage.removeItem('lastUIUpdateTimestamp');
                
                // Finally close the modal after storage is cleared
                // This sequence ensures we don't get flickering from localStorage checks
                setIsUploading(false);
              }, completionDelay);
            }
          } catch (error) {
            console.error('Error parsing progress data:', error);
          }
        };
        
        // Create an error handler function that we can reuse
        const handleError = (event: Event) => {
          console.log('EventSource error, attempting to reconnect...');
          
          // Don't immediately close and end the upload - give it a chance to recover
          // We'll use a timeout to check if we can reconnect
          setTimeout(() => {
            // Make sure we have the global tracking object
            if (!window._activeEventSources) {
              window._activeEventSources = {};
            }
            
            // Get the current EventSource
            const currentSource = window._activeEventSources?.[activeSessionId];
            if (!currentSource) {
              // If we no longer have an EventSource, create a new one
              createNewEventSource();
              return;
            }
            
            // If the connection is in a CLOSED state, try to reopen it
            if (currentSource.readyState === EventSource.CLOSED) {
              console.log('EventSource connection closed, reconnecting...');
              createNewEventSource();
            } else if (currentSource.readyState === EventSource.OPEN) {
              // If it's already reconnected, do nothing
              console.log('EventSource connection recovered');
            } else {
              // Try a final reconnect
              createNewEventSource();
              
              // Set a timeout to check if the reconnection worked
              setTimeout(() => {
                const source = window._activeEventSources?.[activeSessionId];
                if (!source || source.readyState !== EventSource.OPEN) {
                  console.log('EventSource failed to reconnect, closing upload modal');
                  if (source) source.close();
                  if (window._activeEventSources) {
                    delete window._activeEventSources[activeSessionId];
                  }
                  setIsUploading(false);
                }
              }, 3000); // Give it 3 more seconds to connect
            }
          }, 2000);  // Give it 2 seconds before first reconnect attempt
        };
        
        // Function to create a new EventSource with proper setup
        const createNewEventSource = () => {
          // Close any existing source first
          if (window._activeEventSources?.[activeSessionId]) {
            try {
              window._activeEventSources[activeSessionId].close();
            } catch (e) {
              // Ignore errors on close
            }
          }
          
          // Create the global tracking object if it doesn't exist
          if (!window._activeEventSources) {
            window._activeEventSources = {};
          }
          
          // Create new EventSource
          const newSource = new EventSource(`/api/upload-progress/${activeSessionId}`);
          
          // Set event handlers
          newSource.onmessage = handleMessage;
          newSource.onerror = handleError;
          
          // Store in the global registry
          window._activeEventSources[activeSessionId] = newSource;
          
          return newSource;
        };
        
        // Initialize the EventSource
        createNewEventSource();
        
        return;
      }
      
      // Fall back to localStorage check if no active uploads in database
      // But ONLY if there is local storage data (meaning we need to persist it)
      const isUploadingFromStorage = localStorage.getItem('isUploading') === 'true';
      if (isUploadingFromStorage && storedProgress) {
        // Check if the stored progress is fresh (less than 30 minutes old, increased from 5 minutes)
        try {
          const parsedProgress = JSON.parse(storedProgress);
          const savedAt = parsedProgress.savedAt || 0;
          // Use a longer timeout (30 minutes) to ensure modal doesn't disappear too quickly
          const thirtyMinutesAgo = Date.now() - (30 * 60 * 1000);
          
          // If data is stale (more than 30 minutes old), we'll ignore it
          if (savedAt < thirtyMinutesAgo) {
            console.log('Stored upload progress is stale (older than 30 minutes), clearing local storage');
            localStorage.removeItem('isUploading');
            localStorage.removeItem('uploadProgress');
            localStorage.removeItem('uploadSessionId');
            return;
          }
          
          // Enhanced error detection: Check if the upload has completed or has an error
          const stageLower = parsedProgress.stage?.toLowerCase() || '';
          const isCompleteOrError = stageLower.includes('complete') || 
                                    stageLower.includes('error') || 
                                    stageLower.includes('cancelled');
                                    
          // Also check for error-like conditions that should be cleared
          const isErrorCondition = stageLower === 'error' || 
                                   parsedProgress.total === 0 ||
                                   parsedProgress.error;
                                   
          if (isCompleteOrError || isErrorCondition) {
            console.log('Found completed or error upload in localStorage, clearing data');
            // Clear all localStorage items related to uploads
            localStorage.removeItem('isUploading');
            localStorage.removeItem('uploadProgress');
            localStorage.removeItem('uploadSessionId');
            localStorage.removeItem('lastProgressTimestamp');
            localStorage.removeItem('lastUIUpdateTimestamp');
            localStorage.removeItem('lastDisplayTime');
            return;
          }
          
          console.log('Using localStorage data since database check returned no active sessions');
          // We've already parsed the progress, so use it directly
          setUploadProgress(parsedProgress);
          setIsUploading(true);
          
          // Set a longer display time to ensure modal doesn't disappear too quickly
          localStorage.setItem('lastDisplayTime', Date.now().toString());
          
          // Check if there's an active session ID from localStorage
          const sessionId = getCurrentUploadSessionId();
          if (sessionId) {
            console.log('Reconnecting to upload session from localStorage:', sessionId);
            
            // Set up a more robust EventSource for progress updates
            // Keep a reference for cleanup and reconnection
            // Create a message handler function that we can reuse
            const handleMessage = (event: MessageEvent) => {
              try {
                // Make sure we have valid JSON data
                if (!event.data) {
                  return;
                }
                
                // Parse the progress data
                const progress = JSON.parse(event.data);
                
                // Validate that we have a valid progress object
                if (!progress || typeof progress !== 'object') {
                  return;
                }
                
                // Log the progress data for debugging
                console.log("Progress event received:", progress);
                
                // Important: Check timestamps - only update if this is newer than our last update
                // This prevents out-of-order updates from causing flickering
                const currentTimestamp = progress.timestamp || Date.now();
                const lastTimestamp = parseInt(localStorage.getItem('lastProgressTimestamp') || '0');
                
                // Anti-flicker protection
                const lastUIUpdate = parseInt(localStorage.getItem('lastUIUpdateTimestamp') || '0');
                const now = Date.now();
                const timeSinceLastUpdate = now - lastUIUpdate;
                
                // Only update if this is a newer message
                if (currentTimestamp >= lastTimestamp) {
                  // Store the latest progress in localStorage immediately
                  // This ensures we don't lose data even if UI updates are debounced
                  localStorage.setItem('uploadProgress', JSON.stringify({
                    ...progress,
                    savedAt: now
                  }));
                  localStorage.setItem('lastProgressTimestamp', currentTimestamp.toString());
                  
                  // DEBOUNCE UI updates - limit to one update per 300ms to prevent flickering
                  // BUT make exceptions for important state changes that should be immediate
                  const isImportantStateChange = 
                    // Always show immediately when batch changes
                    progress.batchNumber !== uploadProgress.batchNumber ||
                    // Always show immediately when state changes (complete, error, etc)
                    progress.stage !== uploadProgress.stage ||
                    // Always update if we haven't updated in over 500ms
                    timeSinceLastUpdate > 500;
                    
                  if (isImportantStateChange || timeSinceLastUpdate > 300) {
                    // Log what we're sending to the UI
                    console.log("Progress being sent to UI:", progress);
                    
                    // Update UI with progress
                    setUploadProgress(progress);
                    
                    // Record the time of this UI update
                    localStorage.setItem('lastUIUpdateTimestamp', now.toString());
                  }
                }
                
                // Check for completion states in the stage message
                const stageLower = progress.stage?.toLowerCase() || '';
                const isComplete = stageLower.includes('complete');
                const isError = stageLower.includes('error');
                const isCancelled = stageLower.includes('cancelled');
                
                // If the upload is complete or has an error, close the connection
                if (isComplete || isError || isCancelled) {
                  // Check if eventSource still exists before closing
                  if (window._activeEventSources?.[sessionId]) {
                    try {
                      window._activeEventSources[sessionId].close();
                      if (window._activeEventSources) {
                        delete window._activeEventSources[sessionId];
                      }
                    } catch (e) {
                      // Suppress error
                    }
                  }
                  
                  // If it completed successfully, refresh data to show new records
                  if (isComplete) {
                    // First refresh data, but wait a bit for server to finalize everything
                    setTimeout(() => {
                      refreshData();
                    }, 500);
                  }
                  
                  // Use a timer before closing the modal to prevent abrupt UI changes
                  // This gives user time to see the final status
                  const completionDelay = isError ? 2000 : 3000;
                  
                  // For all completion states, we'll delay the modal close slightly
                  // This prevents abrupt UI changes and makes the experience smoother
                  setTimeout(() => {
                    // Clear the upload progress from localStorage
                    localStorage.removeItem('isUploading');
                    localStorage.removeItem('uploadProgress');
                    localStorage.removeItem('uploadSessionId');
                    localStorage.removeItem('lastProgressTimestamp');
                    localStorage.removeItem('lastUIUpdateTimestamp');
                    
                    // Finally close the modal after storage is cleared
                    setIsUploading(false);
                    
                    // Show a toast or alert to inform the user if there was an error
                    if (isError) {
                      console.error('Upload failed with error:', progress.stage);
                    }
                  }, completionDelay);
                }
              } catch (error) {
                console.error('Error parsing progress data:', error);
              }
            };
            
            // Create an error handler function that we can reuse
            const handleError = (event: Event) => {
              console.log('EventSource error, attempting to reconnect...');
              
              // Store error occurrence timestamp
              const errorTime = new Date();
              
              // Don't immediately close and end the upload - give it a chance to recover
              // We'll use a timeout to check if we can reconnect
              setTimeout(() => {
                try {
                  // Make sure we have the global tracking object
                  if (!window._activeEventSources) {
                    window._activeEventSources = {};
                  }
                  
                  // Get the current EventSource
                  const currentSource = window._activeEventSources?.[sessionId];
                  if (!currentSource) {
                    // If we no longer have an EventSource, create a new one
                    console.log('No active EventSource found, creating new one');
                    createNewEventSource();
                    return;
                  }
                  
                  // If the connection is in a CLOSED state, try to reopen it
                  if (currentSource.readyState === EventSource.CLOSED) {
                    console.log('EventSource connection closed, reconnecting...');
                    createNewEventSource();
                  } else if (currentSource.readyState === EventSource.OPEN) {
                    // If it's already reconnected, do nothing
                    console.log('EventSource connection recovered on its own');
                  } else {
                    // Try a final reconnect
                    console.log('EventSource in connecting state, trying a fresh connection');
                    createNewEventSource();
                    
                    // Set a timeout to check if the reconnection worked
                    setTimeout(() => {
                      try {
                        const source = window._activeEventSources?.[sessionId];
                        if (!source || source.readyState !== EventSource.OPEN) {
                          console.log('EventSource failed to reconnect after multiple attempts, closing upload modal');
                          if (source) {
                            try {
                              source.close();
                            } catch (closeError) {
                              console.error('Error closing EventSource:', closeError);
                            }
                          }
                          
                          // Clean up the reference
                          if (window._activeEventSources) {
                            delete window._activeEventSources[sessionId];
                          }
                          
                          // Show error to user and reset upload state
                          setIsUploading(false);
                          
                          // Clear stored upload data
                          localStorage.removeItem('isUploading');
                          localStorage.removeItem('uploadProgress');
                          localStorage.removeItem('uploadSessionId');
                          
                          // We could show a toast notification here about connection issues
                        } else {
                          console.log('EventSource reconnected successfully');
                        }
                      } catch (innerError) {
                        console.error('Error during reconnection check:', innerError);
                        // Failsafe: reset upload state
                        setIsUploading(false);
                      }
                    }, 5000); // Give it 5 seconds to connect
                  }
                } catch (outerError) {
                  console.error('Error in EventSource reconnection logic:', outerError);
                  // Failsafe: reset upload state on any error in the reconnection logic
                  setIsUploading(false);
                }
              }, 2000);  // Give it 2 seconds before first reconnect attempt
            };
            
            // Function to create a new EventSource with proper setup
            const createNewEventSource = () => {
              // Close any existing source first
              if (window._activeEventSources?.[sessionId]) {
                try {
                  window._activeEventSources[sessionId].close();
                } catch (e) {
                  // Ignore errors on close
                }
              }
              
              // Create the global tracking object if it doesn't exist
              if (!window._activeEventSources) {
                window._activeEventSources = {};
              }
              
              // Create new EventSource
              const newSource = new EventSource(`/api/upload-progress/${sessionId}`);
              
              // Set event handlers
              newSource.onmessage = handleMessage;
              newSource.onerror = handleError;
              
              // Store in the global registry
              window._activeEventSources[sessionId] = newSource;
              
              return newSource;
            };
            
            // Initialize the EventSource
            createNewEventSource();
          } else {
            // No active session found, reset the upload state
            setIsUploading(false);
          }
        } catch (error) {
          // Error silently handled - removed console.error
          setIsUploading(false);
        }
      }
    } catch (error) {
      // Error silently handled - removed console.error
      setIsUploading(false);
    }
  };

  // NOW: ONLY CHECK ONCE PER PAGE LOAD/REFRESH, NOT ON ROUTE CHANGES
  // This dramatically reduces API calls and prevents UI flickering
  useEffect(() => {
    // Set a flag in sessionStorage to track if we've already checked this page load
    const hasCheckedThisPageLoad = sessionStorage.getItem('checkedActiveUploadsOnLoad');
    
    // Start with sessionId from localStorage to reduce flickering
    const storedSessionId = localStorage.getItem('uploadSessionId');
    const storedIsUploading = localStorage.getItem('isUploading') === 'true';
    
    // If we already checked this page load OR we're already uploading, don't check again
    if (hasCheckedThisPageLoad === 'true' || storedIsUploading) {
      console.log('⏭️ Skipping database check - already checked this page load or already uploading');
      return;
    }
    
    // Mark that we've checked for this page load
    sessionStorage.setItem('checkedActiveUploadsOnLoad', 'true');
    
    console.log('🔍 ONE-TIME CHECK for active uploads on route:', location);
    
    // Make a single check for active uploads - this will only happen ONCE per page load/refresh
    checkAndReconnectToActiveUploads();
    
    // NO POLLING - we only check once
    // This is a major optimization that dramatically reduces database load
    
    // No need for cleanup since we're not setting any intervals
  }, [location]);

  // WebSocket setup for all non-upload messages (like feedback, post updates)
  // We'll keep this separate from the upload progress handling to avoid conflicts
  useEffect(() => {
    // Flag to determine if we're in upload mode - only handle WebSocket progress in this mode
    // We'll use EventSource for the main method of communication during uploads
    const isInUploadMode = localStorage.getItem('isUploading') === 'true';

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const socket = new WebSocket(wsUrl);

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle sentiment feedback updates (for real-time UI updates)
        if (data.type === 'feedback-update') {
          // This will trigger a refresh of all data including sentiment posts
          refreshData();
          
          // Display a toast notification about the sentiment update
          toast({
            title: 'Model Updated',
            description: `Sentiment model has been trained with new feedback.`,
            variant: 'default',
          });
        }
        // Handle specific post update messages
        else if (data.type === 'post-updated') {
          // Force an immediate refresh to update the UI with the new sentiment
          refreshData();
          
          // Display a toast notification about the post update
          toast({
            title: 'Post Updated',
            description: `Sentiment has been updated successfully.`,
            variant: 'default',
          });
        }
        // We're now IGNORING progress updates from WebSocket when using EventSource
        // This prevents duplicate updates that cause flickering
      } catch (error) {
        // Error silently handled
      }
    };

    return () => {
      socket.close();
    };
  }, []);

  // This section has been moved after the refreshData function

  // Calculate stats with safety checks for array data
  const activeDiastersCount = Array.isArray(disasterEvents) ? disasterEvents.length : 0;
  const analyzedPostsCount = Array.isArray(sentimentPosts) ? sentimentPosts.length : 0;

  // Calculate dominant sentiment with proper array check and percentages
  const sentimentCounts: Record<string, number> = {};
  const totalPosts = Array.isArray(sentimentPosts) ? sentimentPosts.length : 0;
  
  if (totalPosts > 0) {
    sentimentPosts.forEach((post: SentimentPost) => {
      sentimentCounts[post.sentiment] = (sentimentCounts[post.sentiment] || 0) + 1;
    });
  }

  // Sort sentiments by count from highest to lowest
  const sortedSentiments = Object.entries(sentimentCounts)
    .sort((a, b) => b[1] - a[1]);
    
  // Get the dominant sentiment
  const dominantSentiment = sortedSentiments.length > 0 
    ? sortedSentiments[0]?.[0] 
    : "Neutral";
    
  // Calculate percentages for each sentiment
  const sentimentPercentages = Object.fromEntries(
    Object.entries(sentimentCounts).map(([sentiment, count]) => [
      sentiment, 
      totalPosts > 0 ? Math.round((count / totalPosts) * 100) : 0
    ])
  );
  
  // Calculate dominant sentiment percentage
  const dominantSentimentPercentage = sentimentPercentages[dominantSentiment] || 0;
  
  // Calculate second most dominant sentiment if available
  const secondDominantSentiment = sortedSentiments.length > 1 
    ? sortedSentiments[1]?.[0] 
    : null;
  const secondDominantSentimentPercentage = secondDominantSentiment 
    ? sentimentPercentages[secondDominantSentiment] 
    : 0;
    
  // Calculate dominant disaster type with proper array check
  const disasterCounts: Record<string, number> = {};
  let validDisasterPostsCount = 0;
  
  if (totalPosts > 0) {
    sentimentPosts.forEach((post: SentimentPost) => {
      if (post.disasterType && 
          post.disasterType !== "Not Specified" && 
          post.disasterType !== "NONE" && 
          post.disasterType !== "None" && 
          post.disasterType !== "null" && 
          post.disasterType !== "undefined") {
        disasterCounts[post.disasterType] = (disasterCounts[post.disasterType] || 0) + 1;
        validDisasterPostsCount++;
      }
    });
  }
  
  // Sort disaster types by count from highest to lowest
  const sortedDisasters = Object.entries(disasterCounts)
    .sort((a, b) => b[1] - a[1]);
  
  // Get the dominant disaster
  const dominantDisaster = sortedDisasters.length > 0 
    ? sortedDisasters[0]?.[0] 
    : "Unknown";
    
  // Calculate percentages for each disaster type
  const disasterPercentages = Object.fromEntries(
    Object.entries(disasterCounts).map(([disasterType, count]) => [
      disasterType, 
      validDisasterPostsCount > 0 ? Math.round((count / validDisasterPostsCount) * 100) : 0
    ])
  );
  
  // Calculate dominant disaster percentage
  const dominantDisasterPercentage = disasterPercentages[dominantDisaster] || 0;
  
  // Calculate second most dominant disaster if available
  const secondDominantDisaster = sortedDisasters.length > 1 
    ? sortedDisasters[1]?.[0] 
    : null;
  const secondDominantDisasterPercentage = secondDominantDisaster 
    ? disasterPercentages[secondDominantDisaster] 
    : 0;

  // Calculate average model confidence with safety checks
  const totalConfidence = Array.isArray(sentimentPosts) 
    ? sentimentPosts.reduce((sum: number, post: SentimentPost) => sum + (post.confidence || 0), 0)
    : 0;
  const modelConfidence = Array.isArray(sentimentPosts) && sentimentPosts.length > 0 
    ? totalConfidence / sentimentPosts.length 
    : 0;

  return (
    <DisasterContext.Provider
      value={{
        sentimentPosts,
        disasterEvents,
        analyzedFiles,
        isLoadingSentimentPosts,
        isLoadingDisasterEvents,
        isLoadingAnalyzedFiles,
        isUploading,
        uploadProgress,
        errorSentimentPosts: errorSentimentPosts as Error | null,
        errorDisasterEvents: errorDisasterEvents as Error | null,
        errorAnalyzedFiles: errorAnalyzedFiles as Error | null,
        activeDiastersCount,
        analyzedPostsCount,
        dominantSentiment,
        dominantDisaster,
        modelConfidence,
        // Sentiment statistics
        dominantSentimentPercentage,
        secondDominantSentiment,
        secondDominantSentimentPercentage,
        sentimentPercentages,
        // Disaster statistics
        dominantDisasterPercentage,
        secondDominantDisaster,
        secondDominantDisasterPercentage,
        disasterPercentages,
        // Filters
        selectedDisasterType,
        setSelectedDisasterType,
        setIsUploading,
        setUploadProgress,
        refreshData
      }}
    >
      {children}
    </DisasterContext.Provider>
  );
}

export function useDisasterContext() {
  const context = useContext(DisasterContext);
  if (context === undefined) {
    throw new Error("useDisasterContext must be used within a DisasterContextProvider");
  }
  return context;
}