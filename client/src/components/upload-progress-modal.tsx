import { motion } from "framer-motion";
import { 
  CheckCircle, 
  Clock, 
  Database, 
  FileText, 
  Loader2, 
  XCircle,
  AlertCircle,
  BarChart3,
  Server,
  Terminal
} from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { createPortal } from "react-dom";
import React, { useEffect, useState, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { ToastAction } from "@/components/ui/toast";

// Create a BroadcastChannel for cross-tab communication
const uploadBroadcastChannel = typeof window !== 'undefined' ? new BroadcastChannel('upload_status') : null;
import { cancelUpload, getCurrentUploadSessionId } from "@/lib/api";

export function UploadProgressModal() {
  const { isUploading, uploadProgress, setIsUploading, setUploadProgress } = useDisasterContext();
  const [isCancelling, setIsCancelling] = useState(false);
  const [showCancelDialog, setShowCancelDialog] = useState(false);
  const { toast } = useToast(); // Initialize toast hook
  
  // Define a clean-up function that will be used for both direct modal closing and broadcast handling
  const cleanupAndClose = React.useCallback(() => {
    console.log('🧹 Cleanup and close function triggered');
    
    // Clear all localStorage items
    localStorage.removeItem('isUploading');
    localStorage.removeItem('uploadProgress');
    localStorage.removeItem('uploadSessionId');
    localStorage.removeItem('lastProgressTimestamp');
    localStorage.removeItem('lastDatabaseCheck');
    localStorage.removeItem('serverRestartProtection');
    localStorage.removeItem('serverRestartTimestamp');
    localStorage.removeItem('uploadCompleteBroadcasted');
    localStorage.removeItem('lastUIUpdateTimestamp');
    localStorage.removeItem('uploadStartTime');
    localStorage.removeItem('batchStats');
    
    // Clear the new upload completion markers
    localStorage.removeItem('uploadCompleted');
    localStorage.removeItem('uploadCompletedTimestamp');
    
    // Broadcast cleanup to other tabs
    try {
      if (uploadBroadcastChannel) {
        console.log('📢 Broadcasting cleanup to all tabs');
        uploadBroadcastChannel.postMessage({
          type: 'upload_cleanup',
          timestamp: Date.now()
        });
      }
    } catch (e) {
      console.error('Error broadcasting cleanup:', e);
    }
    
    // Clean up any existing EventSource connections
    if (window._activeEventSources) {
      Object.values(window._activeEventSources).forEach(source => {
        try {
          source.close();
        } catch (e) {
          // Ignore errors on close
        }
      });
      // Reset the collection
      window._activeEventSources = {};
    }
    
    // Update context state
    setIsUploading(false);
    setIsCancelling(false);
    
    console.log('🧹 MODAL CLOSED - ALL LOCALSTORAGE CLEARED');
  }, [setIsUploading, setIsCancelling]);
  
  // Set up broadcast channel listener to handle messages from other tabs
  useEffect(() => {
    if (!uploadBroadcastChannel) return;
    
    // Before setting up listeners, check if another tab already marked completion
    const alreadyCompleted = localStorage.getItem('uploadCompleted') === 'true';
    if (alreadyCompleted && isUploading) {
      console.log('🔄 Another tab already completed the upload, closing this modal');
      // Use a short delay to allow UI to update first
      setTimeout(() => {
        cleanupAndClose();
      }, 500);
    }
    
    // Create completion-specific channel 
    let completionChannel: BroadcastChannel | null = null;
    try {
      completionChannel = new BroadcastChannel('upload_completion');
    } catch (e) {
      console.error('Failed to create completion channel:', e);
    }
    
    const handleCompletionMessage = (event: MessageEvent) => {
      console.log('🏁 Received dedicated completion message:', event.data.type);
      
      if (event.data.type === 'analysis_complete') {
        console.log('🏁 Received DEDICATED completion message - highest priority!');
        
        // Mark as complete in localStorage
        localStorage.setItem('uploadCompleted', 'true');
        localStorage.setItem('uploadCompletedTimestamp', Date.now().toString());
        
        // Force showing the completion state
        setUploadProgress({
          ...uploadProgress,
          stage: 'Analysis complete',
          processed: uploadProgress.total || 10,
          total: uploadProgress.total || 10,
          currentSpeed: 0,
          timeRemaining: 0
        });
        
        // Auto-close after a delay
        setTimeout(() => {
          console.log('⏰ AUTO-CLOSE TRIGGERED BY DEDICATED COMPLETION CHANNEL');
          cleanupAndClose();
        }, 3000);
      }
    };
    
    const handleBroadcastMessage = (event: MessageEvent) => {
      console.log('📻 Upload modal received broadcast:', event.data.type);
      
      // Handle completion messages from other tabs
      if (event.data.type === 'upload_complete') {
        console.log('📊 Received completion message from another tab');
        
        // Force this tab to show the upload modal if we're not already showing it
        if (!isUploading) {
          console.log('🔄 This tab was not showing the upload modal, forcing it to show');
          setIsUploading(true);
        }
        
        // First set stage to Analysis Complete
        if (event.data.progress) {
          console.log('📊 Setting progress to Analysis Complete from broadcast');
          
          const updatedProgress = {
            ...event.data.progress,
            stage: 'Analysis complete',
            processed: event.data.progress.total || 10, // Make sure processed equals total
            currentSpeed: 0, // Reset speed
            timeRemaining: 0 // No time remaining
          };
          
          // Update our state
          setUploadProgress(updatedProgress);
          
          // Also update localStorage with the new state
          localStorage.setItem('uploadProgress', JSON.stringify({
            ...updatedProgress,
            savedAt: Date.now()
          }));
          
          // Mark as complete in localStorage for all tabs
          localStorage.setItem('uploadCompleted', 'true');
          localStorage.setItem('uploadCompletedTimestamp', Date.now().toString());
          
          // Add a mark in localStorage to indicate we've shown the completion state
          localStorage.setItem('uploadCompleteBroadcasted', 'true');
          
          // Set a timer to auto-close like the original tab does
          const completionDelay = 3000; // 3 seconds
          console.log(`📊 Will auto-close after ${completionDelay}ms due to broadcast`);
          
          setTimeout(() => {
            console.log('⏰ AUTO-CLOSE TRIGGERED BY BROADCAST MESSAGE');
            cleanupAndClose();
          }, completionDelay);
        }
      }
      
      // Handle force cancel messages from other tabs
      if (event.data.type === 'upload_force_cancelled') {
        console.log('🔥 FORCE CANCEL MODE ACTIVATED FROM BROADCAST');
        cleanupAndClose();
      }
      
      // Handle cleanup messages from other tabs
      if (event.data.type === 'upload_cleanup') {
        console.log('🧹 CLEANUP MESSAGE RECEIVED FROM ANOTHER TAB');
        if (isUploading) {
          console.log('Closing this modal due to cleanup message from another tab');
          cleanupAndClose();
        }
      }
    };
    
    // Add listeners to both channels
    uploadBroadcastChannel.addEventListener('message', handleBroadcastMessage);
    if (completionChannel) {
      completionChannel.addEventListener('message', handleCompletionMessage);
    }
    
    return () => {
      uploadBroadcastChannel.removeEventListener('message', handleBroadcastMessage);
      if (completionChannel) {
        completionChannel.removeEventListener('message', handleCompletionMessage);
        completionChannel.close();
      }
    };
  }, [setUploadProgress, cleanupAndClose, isUploading, setIsUploading, uploadProgress]);
  
  // Check for server restart protection flag
  // Only consider server restart protection if explicitly set
  // Check for both server restart AND completion status - never show server restart mode for completed uploads
  const serverRestartDetected = localStorage.getItem('serverRestartProtection') === 'true';
  const serverRestartTime = localStorage.getItem('serverRestartTimestamp');
  const uploadCompleted = localStorage.getItem('uploadCompleted') === 'true';
  
  // If upload is completed, ignore server restart protection completely
  if (uploadCompleted && serverRestartDetected) {
    console.log('🏁 Upload completed flag detected - ignoring server restart protection');
    localStorage.removeItem('serverRestartProtection');
    localStorage.removeItem('serverRestartTimestamp');
  }
  
  // We'll determine if we should show server restart mode after we know what stage we're in
  // This will be set after we have the stage value
  
  // Regular check with database boss - if boss says no active sessions but we're showing
  // a modal, boss is right and we should close the modal
  // BUT THIS HAS A SPECIAL EXCEPTION FOR SERVER RESTARTS
  // We need to place this functionality AFTER all variable declarations
  
  useEffect(() => {
    if (!isUploading) return; // No need to check if we're not showing a modal
    
    // Function to verify with database - BUT LOCAL STORAGE IS THE TRUE BOSS NOW!
    const checkWithDatabaseBoss = async () => {
      try {
        // PRIORITY FOR VISIBILITY: LOCAL STORAGE > DATABASE
        // First let's verify our localStorage state to make sure we should still show modal
        const storageExpirationMinutes = 30; // EXTENDED from 2 to 30 minutes!
        const storedUploadProgress = localStorage.getItem('uploadProgress');
        const storedSessionId = localStorage.getItem('uploadSessionId');
        const storedIsUploading = localStorage.getItem('isUploading');
        
        // If we have active state in localStorage, KEEP SHOWING even if database says no
        if (storedIsUploading === 'true' || storedSessionId) {
          console.log('🔒 LOCAL STORAGE HAS UPLOAD STATE - KEEPING MODAL VISIBLE', storedSessionId);
          
          // Super strong persistence - force UI to match localStorage state
          if (!isUploading) {
            setIsUploading(true);
          }
          
          // Now check with database - but just for data updates, not for visibility control
          console.log('📊 Checking database for progress updates only (not for visibility control)');
          const response = await fetch('/api/active-upload-session');
        
          if (response.ok) {
            const data = await response.json();
            
            if (data.sessionId) {
              console.log('✅ DATABASE CONFIRMS ACTIVE SESSION', data.sessionId);
              // If we already have sessionId in localStorage, update it if different
              if (storedSessionId !== data.sessionId) {
                localStorage.setItem('uploadSessionId', data.sessionId);
              }
            }
          }
        } else {
          // No upload state in localStorage, ask database
          console.log('📊 Upload modal checking with database');
          const response = await fetch('/api/active-upload-session');
          
          if (response.ok) {
            const data = await response.json();
            
            // If database has active session but we don't know about it yet
            if (data.sessionId) {
              console.log('👑 DATABASE HAS ACTIVE SESSION WE DIDNT KNOW ABOUT', data.sessionId);
              
              // Update localStorage with the session ID
              localStorage.setItem('uploadSessionId', data.sessionId);
              localStorage.setItem('isUploading', 'true');
              
              // Show the upload modal
              setIsUploading(true);
            }
            // Otherwise, both localStorage and database agree - no active uploads
          }
        }
      } catch (error) {
        // Silently handle errors to avoid disrupting the UI
        console.error('Error checking database:', error);
      }
    };
    
    // Check with database boss immediately
    checkWithDatabaseBoss();
    
    // Then set up regular checks with the boss
    const intervalId = setInterval(checkWithDatabaseBoss, 5000);
    
    // Cleanup on unmount
    return () => clearInterval(intervalId);
  }, [isUploading, setIsUploading]);
  
  // Poll the specialized API endpoint for completion status
  useEffect(() => {
    if (!isUploading) return; // No need to check if not uploading
    
    const checkCompletionStatus = async () => {
      try {
        const response = await fetch('/api/upload-complete-check');
        if (response.ok) {
          const data = await response.json();
          
          // If the server says upload is complete, force completion state for ALL tabs
          if (data.uploadComplete && data.sessionId) {
            console.log('🌟 COMPLETION CHECK API SAYS UPLOAD IS COMPLETE!', data.sessionId);
            
            // Update our state first
            setUploadProgress({
              ...uploadProgress,
              stage: 'Analysis complete',
              processed: uploadProgress.total || 100,
              total: uploadProgress.total || 100,
              currentSpeed: 0,
              timeRemaining: 0
            });
            
            // Save to localStorage for other tabs
            localStorage.setItem('uploadCompleted', 'true');
            localStorage.setItem('uploadCompletedTimestamp', Date.now().toString());
            
            // Also broadcast to all tabs
            try {
              // Use both broadcast channels for notification
              if (uploadBroadcastChannel) {
                uploadBroadcastChannel.postMessage({
                  type: 'upload_complete',
                  progress: {
                    ...uploadProgress,
                    stage: 'Analysis complete',
                    processed: uploadProgress.total || 100,
                    total: uploadProgress.total || 100,
                    currentSpeed: 0,
                    timeRemaining: 0
                  },
                  timestamp: Date.now()
                });
              }
              
              // Use dedicated completion channel
              const completionChannel = new BroadcastChannel('upload_completion');
              completionChannel.postMessage({
                type: 'analysis_complete',
                timestamp: Date.now()
              });
              setTimeout(() => {
                try { completionChannel.close(); } catch (e) { /* ignore */ }
              }, 1000);
              
            } catch (e) {
              console.error('Error broadcasting via completion check:', e);
            }
            
            // Set a timer to auto-close this tab too
            setTimeout(() => {
              console.log('⏰ AUTO-CLOSE TRIGGERED BY COMPLETION CHECK API');
              cleanupAndClose();
            }, 3000);
          }
        }
      } catch (error) {
        console.error('Error checking upload completion status:', error);
      }
    };
    
    // Check immediately on mount
    checkCompletionStatus();
    
    // Check every 1 second - this is specifically designed to catch completion
    // events and has a high frequency because completion needs immediate response
    const intervalId = setInterval(checkCompletionStatus, 1000);
    
    return () => clearInterval(intervalId);
  }, [isUploading, uploadProgress, setUploadProgress, cleanupAndClose]);

  // No local forceCloseModal function anymore - we use the memoized version forceCloseModalMemo
  
  // Handle cancel button click with force option
  const handleCancel = async (forceCancel = false) => {
    if (isCancelling) return;
    
    setShowCancelDialog(false);
    setIsCancelling(true);
    try {
      const result = await cancelUpload(forceCancel);
      
      if (result.success) {
        // Force close the modal instead of waiting for events
        cleanupAndClose();
      } else {
        // If normal cancel failed, show option for force cancel
        if (!forceCancel) {
          toast({
            title: 'Cancel Failed',
            description: 'Server could not cancel the upload. Try Force Cancel instead.',
            variant: 'destructive',
            action: (
              <ToastAction 
                altText="Force Cancel" 
                onClick={() => handleCancel(true)}
              >
                Force Cancel
              </ToastAction>
            ),
          });
        } else {
          // Even if server force cancel failed, still close UI
          cleanupAndClose();
          toast({
            title: 'Force Canceled',
            description: 'The upload has been forcefully canceled in this browser tab.',
            variant: 'destructive',
          });
        }
        setIsCancelling(false);
      }
    } catch (error) {
      console.error('Error cancelling upload:', error);
      
      // On force cancel, always close the modal even if there was an error
      if (forceCancel) {
        cleanupAndClose();
        toast({
          title: 'Force Canceled',
          description: 'The upload has been forcefully canceled in this browser tab.',
          variant: 'destructive',
        });
      } else {
        // For regular cancel errors, offer force cancel option
        toast({
          title: 'Error',
          description: 'Failed to cancel. Try Force Cancel instead.',
          variant: 'destructive',
          action: (
            <ToastAction 
              altText="Force Cancel" 
              onClick={() => handleCancel(true)}
            >
              Force Cancel
            </ToastAction>
          ),
        });
        setIsCancelling(false);
      }
    }
  };

  // Extract values from uploadProgress
  const { 
    stage = 'Processing...', 
    processed: rawProcessed = 0, 
    total = 100,
    processingStats = {
      successCount: 0,
      errorCount: 0,
      averageSpeed: 0
    },
    batchNumber = 0,
    totalBatches = 0,
    batchProgress = 0,
    currentSpeed = 0,
    timeRemaining = 0,
    error = '',
    autoCloseDelay = 3000 // Default to 3 seconds for auto-close
  } = uploadProgress;
  
  // Add auto-close timer for both "Analysis complete" and error states
  // Memoize forceCloseModal to prevent unnecessary re-renders
  const forceCloseModalMemo = React.useCallback(async () => {
    // Set local cancelling state immediately to prevent multiple calls
    setIsCancelling(true);
    
    try {
      // First try to cancel the upload through the API to ensure database cleanup
      // This ensures the server-side cleanup occurs, deleting the session from the database
      const sessionId = localStorage.getItem('uploadSessionId');
      
      // Clear all localStorage items FIRST to prevent any race conditions
      localStorage.removeItem('isUploading');
      localStorage.removeItem('uploadProgress');
      localStorage.removeItem('uploadSessionId');
      localStorage.removeItem('lastProgressTimestamp');
      localStorage.removeItem('lastDatabaseCheck');
      localStorage.removeItem('serverRestartProtection');
      localStorage.removeItem('serverRestartTimestamp');
      
      // Also clear any other upload-related items that might be causing persistence
      localStorage.removeItem('lastUIUpdateTimestamp');
      localStorage.removeItem('uploadStartTime');
      localStorage.removeItem('batchStats');
      
      if (sessionId) {
        try {
          // Try to cancel through the API (which will delete the session from the database)
          await cancelUpload();
          console.log('Force close: Upload cancelled through API to ensure database cleanup');
          
          // Make a direct call to cleanup error sessions for immediate cleanup
          await fetch('/api/cleanup-error-sessions', {
            method: 'POST'
          });
          console.log('Force cleanup: Called error session cleanup API');
        } catch (e) {
          console.error('Error during API cleanup (continuing anyway):', e);
        }
      }
    } catch (error) {
      console.error('Error cancelling upload during force close:', error);
    } finally {
      // Double-check that localStorage is really cleared
      localStorage.removeItem('isUploading');
      localStorage.removeItem('uploadProgress');
      localStorage.removeItem('uploadSessionId');
      
      // Clean up any existing EventSource connections
      if (window._activeEventSources) {
        Object.values(window._activeEventSources).forEach(source => {
          try {
            source.close();
          } catch (e) {
            // Ignore errors on close
          }
        });
        // Reset the collection
        window._activeEventSources = {};
      }
      
      // Finally update context state - do this AFTER cleanup is complete
      // to prevent race conditions with new session checks
      setIsUploading(false);
      setIsCancelling(false);
      
      // Also broadcast the cancellation to all tabs
      try {
        if (uploadBroadcastChannel) {
          console.log('📢 Broadcasting upload cancelled to all tabs');
          uploadBroadcastChannel.postMessage({
            type: 'upload_force_cancelled',
            timestamp: Date.now()
          });
          
          // Use a second channel object to ensure delivery in case the first is closed
          const backupChannel = new BroadcastChannel('upload_status');
          backupChannel.postMessage({
            type: 'upload_force_cancelled',
            timestamp: Date.now()
          });
          
          // Close the backup channel after a delay
          setTimeout(() => {
            try { backupChannel.close(); } catch (e) { /* ignore */ }
          }, 1000);
        }
      } catch (e) {
        console.error('Error broadcasting cancellation:', e);
      }
      
      console.log('🧹 MODAL FORCED CLOSED - ALL LOCALSTORAGE CLEARED');
    }
  }, [setIsUploading, setIsCancelling]);

  useEffect(() => {
    // INSTANT CLOSE FOR ERRORS, brief delay for completion
    if (isUploading) {
      if (stage === 'Upload Error') {
        // INSTANT CLEANUP FOR ERRORS - close immediately with tiny delay
        console.log(`🚨 ERROR DETECTED - CLOSING IMMEDIATELY`);
        
        // Use minimal delay (50ms) just to ensure state can be seen
        const closeTimerId = setTimeout(() => {
          console.log(`⏰ ERROR AUTO-CLOSE TRIGGERED IMMEDIATELY`);
          
          // Ultra-aggressive cleanup for error states
          for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key) localStorage.removeItem(key); // Clear EVERYTHING
          }
          
          forceCloseModalMemo(); // Close the modal automatically
        }, 50); // Super short delay
        
        return () => clearTimeout(closeTimerId);
      }
      else if (stage === 'Analysis complete') {
        // For completion, show success briefly (3 seconds)
        const completionDelay = autoCloseDelay || 3000; // Default to 3 seconds
        console.log(`🎯 Analysis complete - will auto-close after ${completionDelay}ms`);
        
        // Mark as complete in localStorage - this is critical for cross-tab sync
        localStorage.setItem('uploadCompleted', 'true');
        localStorage.setItem('uploadCompletedTimestamp', Date.now().toString());
        
        // Update localStorage with current state
        localStorage.setItem('uploadProgress', JSON.stringify({
          ...uploadProgress,
          stage: 'Analysis complete',
          processed: total, // Make sure processed == total
          savedAt: Date.now()
        }));
        
        // Broadcast the completion status to all tabs immediately
        try {
          if (uploadBroadcastChannel) {
            console.log('📢 Broadcasting analysis complete to all tabs');
            uploadBroadcastChannel.postMessage({
              type: 'upload_complete',
              progress: {
                ...uploadProgress,
                stage: 'Analysis complete',
                processed: total // Make sure processed equals total for consistency
              },
              timestamp: Date.now()
            });
            
            // Use a second channel object to ensure delivery in case the first is closed
            const backupChannel = new BroadcastChannel('upload_status');
            backupChannel.postMessage({
              type: 'upload_complete',
              progress: {
                ...uploadProgress,
                stage: 'Analysis complete',
                processed: total
              },
              timestamp: Date.now()
            });
            
            // Close the backup channel after a delay
            setTimeout(() => {
              try { backupChannel.close(); } catch (e) { /* ignore */ }
            }, 1000);
            
            // As an extra safeguard, also use a completion-specific channel
            const completionChannel = new BroadcastChannel('upload_completion');
            completionChannel.postMessage({
              type: 'analysis_complete',
              timestamp: Date.now()
            });
            
            // Close the completion channel after a delay
            setTimeout(() => {
              try { completionChannel.close(); } catch (e) { /* ignore */ }
            }, 1000);
          }
        } catch (e) {
          console.error('Error broadcasting completion:', e);
        }
        
        // Add a mark in localStorage to indicate we've shown the completion state
        localStorage.setItem('uploadCompleteBroadcasted', 'true');
        
        const closeTimerId = setTimeout(() => {
          console.log(`⏰ COMPLETION AUTO-CLOSE TRIGGERED AFTER ${completionDelay}ms`);
          cleanupAndClose(); // Close the modal automatically using our unified cleanup function
        }, completionDelay);
        
        return () => {
          console.log('Cleaning up completion timer');
          clearTimeout(closeTimerId);
        };
      }
    }
  }, [isUploading, stage, forceCloseModalMemo, autoCloseDelay]);
  
  // Don't render the modal if not uploading
  if (!isUploading) return null;
  
  // ENHANCED STAGE DETECTION LOGIC
  // Convert stage to lowercase once for all checks
  const stageLower = stage.toLowerCase();
  
  // IMPROVED: Check if we're in the initializing phase with stronger prioritization
  // Include the initial loading state when application is started or refreshed
  // and restoring an in-progress upload
  // Force initialization display until proper processing starts
  const isInitializing = (rawProcessed === 0) || 
                        stageLower.includes('initializing') || 
                        stageLower.includes('loading csv file') ||
                        stageLower.includes('file loaded') ||
                        stageLower.includes('identifying columns') || 
                        stageLower.includes('identified data columns') ||
                        stageLower.includes('preparing') ||
                        stageLower.includes('starting');
                        
  // IMPROVED: Make initialization a higher priority state
  // This ensures the initialization UI is always shown during the early phases
  
  // Keep original server values for display
  const processedCount = rawProcessed;
  
  // Basic state detection - clear, explicit flags
  const isPaused = stageLower.includes('pause between batches');
  const isLoading = stageLower.includes('loading') || stageLower.includes('preparing');
  const isProcessingRecord = stageLower.includes('processing record') || stageLower.includes('completed record');
  
  // Extract cooldown time if present in stage (e.g., '60-second pause between batches: 42 seconds remaining')
  let cooldownTime = null;
  if (isPaused && stageLower.includes('seconds remaining')) {
    const match = stageLower.match(/(\d+)\s+seconds?\s+remaining/);
    if (match && match[1]) {
      cooldownTime = parseInt(match[1], 10);
    }
  }
  
  // Now that we have stageLower, determine if we should show server restart protection
  // Don't show server restart mode during normal batch pauses or when processing is complete
  const serverRestartProtection = serverRestartDetected && 
                                 !(isPaused || stageLower.includes('batch') || 
                                   stageLower.includes('seconds remaining') || 
                                   stageLower.includes('complete') || 
                                   stageLower.includes('analysis complete'));
  
  // Consider any active work state as "processing"
  const isProcessing = isProcessingRecord || isPaused || stageLower.includes('processing');
  
  // Only set complete when explicitly mentioned OR when we've processed everything
  // Improved completion detection with more specific matches
  const isReallyComplete = stageLower.includes('completed all') || 
                        stageLower.includes('analysis complete') || 
                        stageLower.includes('complete') ||
                        stageLower === 'complete' ||
                        (rawProcessed >= total * 0.99 && total > 0);
  
  // FIXED VERSION - SIMPLER LOGIC:
  // 1. If stage includes "complete" or "analysis complete", it's NEVER an error
  // 2. If stage is explicitly "error" or contains "failed", it IS an error
  // 3. If we have an error message but our progress is complete, DON'T show error
  
  const isCompletionState = stageLower.includes('complete') || stageLower.includes('analysis complete');
  const isErrorStage = stageLower === 'error' || stageLower.includes('failed') || stageLower.includes('critical error');
  
  // Simplified error detection
  const hasError = isErrorStage && !isCompletionState && !isReallyComplete;
                 
  // Final completion state - explicitly check if it's not an error state first AND we've processed all (or nearly all) records
  // Only show "Analysis Complete!" when we've processed at least 99% of records
  const isComplete = isReallyComplete && !hasError && 
                    (total > 0 && processedCount >= total * 0.99) &&
                    !serverRestartProtection;
  
  // Calculate completion percentage safely - ensure it's visible when processing
  const percentComplete = total > 0 
    ? Math.min(100, Math.max(isProcessing ? 1 : 0, Math.round((processedCount / total) * 100)))
    : 0;
  
  // Check for cancellation
  const isCancelled = stageLower.includes('cancel');
  
  // Calculate time remaining in human-readable format
  const formatTimeRemaining = (seconds: number): string => {
    if (!seconds || seconds <= 0) return 'calculating...';
    
    // Calculate a realistic time remaining based on processed records and speed
    // This ensures we're showing updated time even if server doesn't update timeRemaining
    let calculatedTimeRemaining = seconds;
    if (currentSpeed > 0 && total > processedCount) {
      // Calculate time remaining based on current speed and records left
      const recordsRemaining = total - processedCount;
      calculatedTimeRemaining = recordsRemaining / currentSpeed;
    }
    
    // Use the smaller of the two values to be more accurate
    // This prevents stale timeRemaining values from server
    const actualSeconds = Math.min(calculatedTimeRemaining, seconds);
    
    // Less than a minute
    if (actualSeconds < 60) return `${Math.ceil(actualSeconds)} sec`;
    
    // Calculate days, hours, minutes, seconds
    const days = Math.floor(actualSeconds / 86400); // 86400 seconds in a day
    const hours = Math.floor((actualSeconds % 86400) / 3600); // 3600 seconds in an hour
    const minutes = Math.floor((actualSeconds % 3600) / 60);
    const remainingSeconds = Math.ceil(actualSeconds % 60);
    
    // Format based on duration
    if (days > 0) {
      // If we have days, show days and hours
      return `${days}d ${hours}h`;
    } else if (hours > 0) {
      // If we have hours, show hours and minutes
      return `${hours}h ${minutes}m`;
    } else {
      // Otherwise just show minutes and seconds
      return `${minutes}m ${remainingSeconds}s`;
    }
  };

  return createPortal(
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="fixed inset-0 flex items-center justify-center z-[9999]"
    >
      {/* Modern blur backdrop */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-600/20 via-indigo-600/10 to-purple-600/20 backdrop-blur-lg"></div>

      {/* Content Container */}
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        transition={{ duration: 0.3, type: "spring", stiffness: 300, damping: 30 }}
        className="relative bg-white/90 dark:bg-gray-900/90 rounded-2xl overflow-hidden w-full max-w-sm mx-4 shadow-2xl border border-white/20 backdrop-blur"
        style={{
          background: "rgba(255, 255, 255, 0.95)",
          boxShadow: "0 10px 40px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(255, 255, 255, 0.1)",
        }}
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 p-5 text-white relative overflow-hidden">
          {/* Background pattern */}
          <div className="absolute inset-0 opacity-20">
            <div className="absolute right-0 top-0 w-32 h-32 bg-white/20 rounded-full -mr-16 -mt-16"></div>
            <div className="absolute left-0 bottom-0 w-24 h-24 bg-white/10 rounded-full -ml-12 -mb-12"></div>
          </div>
          
          {/* Title */}
          <h3 className="text-xl font-bold text-center mb-4 relative">
            {isComplete ? 'Analysis Complete!' : hasError ? 'Upload Error' : `Processing Records`}
          </h3>
          
          {/* Counter with animations */}
          <motion.div 
            className="flex flex-col items-center justify-center relative z-10"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            {isInitializing ? (
              <div className="py-5 flex flex-col items-center justify-center">
                <Loader2 className="h-12 w-12 animate-spin text-white mb-3" />
                <span className="text-lg font-medium text-white">Preparing Your Dataset...</span>
                <span className="text-sm text-white/70 mt-1">Setting up the processing system</span>
                <span className="text-xs text-white/60 mt-1 max-w-[220px] text-center">
                  {stage.includes('Starting') ? 'Counting records and analyzing structure...' : 
                   stage.includes('column') ? 'Identifying data structure and columns...' : 
                   'Loading data and preparing for analysis...'}
                </span>
              </div>
            ) : (
              <div className="flex items-center justify-center">
                <motion.div 
                  className="relative text-center"
                  key={processedCount}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3, type: "spring" }}
                >
                  <div className="flex items-center justify-center">
                    <span className="text-6xl font-bold text-white drop-shadow-sm">{processedCount}</span>
                    <div className="flex flex-col items-start ml-2">
                      <span className="text-xs text-white/70 uppercase tracking-wider">of</span>
                      <span className="text-2xl font-bold text-white/90">{total}</span>
                    </div>
                  </div>
                  <span className="text-sm mt-1 block text-white/80 font-medium uppercase tracking-wider">Records Processed</span>
                </motion.div>
              </div>
            )}
            
            {/* Progress bar */}
            <div className="w-full mt-4 mb-1">
              <div className="h-2 bg-black/10 rounded-full overflow-hidden relative">
                {isInitializing ? (
                  <motion.div
                    className="absolute top-0 left-0 h-full bg-gradient-to-r from-blue-400 via-indigo-500 to-purple-500"
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ 
                      duration: 1.5, 
                      repeat: Infinity, 
                      repeatType: "reverse",
                      ease: "easeInOut"
                    }}
                    style={{ 
                      backgroundSize: '200% 100%',
                      animation: 'gradientShift 2s linear infinite'
                    }}
                  />
                ) : (
                  <motion.div
                    className={`absolute top-0 left-0 h-full ${
                      hasError 
                        ? 'bg-red-500' 
                        : isComplete 
                          ? 'bg-green-500' 
                          : 'bg-gradient-to-r from-blue-400 to-purple-500'
                    }`}
                    initial={{ width: "0%" }}
                    animate={{ width: `${percentComplete}%` }}
                    transition={{ duration: 0.5 }}
                    style={{ 
                      backgroundSize: '200% 100%',
                      animation: isProcessing && !isComplete && !hasError 
                        ? 'gradientShift 2s linear infinite'
                        : 'none'
                    }}
                  />
                )}
              </div>
              <div className="flex justify-between text-xs mt-1">
                <span className="text-white/70">
                  {isInitializing ? 'Initializing...' : `${Math.floor(percentComplete)}%`}
                </span>
                <span className="text-white/70">
                  {!isInitializing && currentSpeed > 0 ? `${currentSpeed.toFixed(1)} records/sec` : ''}
                </span>
              </div>
            </div>
          </motion.div>
        </div>
        
        {/* Body content */}
        <div className="p-5">
          {/* Status cards */}
          <div className="space-y-3">
            {/* Current status */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm border border-gray-100 dark:border-gray-700">
              <div className="flex items-start gap-3">
                <div className={`p-2 rounded-full flex-shrink-0 ${
                  hasError 
                    ? 'bg-red-100 text-red-600' 
                    : isComplete 
                      ? 'bg-green-100 text-green-600' 
                      : 'bg-blue-100 text-blue-600'
                }`}>
                  {serverRestartProtection ? (
                    <Server className="h-5 w-5 text-blue-500" />
                  ) : hasError ? (
                    <AlertCircle className="h-5 w-5" />
                  ) : isComplete ? (
                    <CheckCircle className="h-5 w-5" />
                  ) : (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  )}
                </div>
                <div className="flex-1">
                  <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {serverRestartProtection ? 'Server Restart Mode' : 'Current Status'}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-300 mt-0.5">
                    {serverRestartProtection ? 'Waiting for server to resume...' : stage}
                  </p>
                  {isPaused && !serverRestartProtection && (
                    <p className="text-xs text-amber-600 mt-1 font-medium">
                      {cooldownTime !== null 
                        ? `Cooldown pause: ${cooldownTime} seconds remaining`
                        : "System is paused between batches to prevent overloading"}
                    </p>
                  )}
                  {serverRestartProtection && (
                    <p className="text-xs text-blue-600 mt-1 font-medium">
                      Your upload was preserved during a server interruption
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Processing stats - show either initializing or processing stats */}
            {isInitializing ? (
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm border border-gray-100 dark:border-gray-700">
                  <div className="flex flex-col">
                    <div className="flex items-center gap-1.5 mb-1">
                      <Database className="h-3.5 w-3.5 text-indigo-500" />
                      <span className="text-xs font-medium text-gray-700 dark:text-gray-300">Status</span>
                    </div>
                    <span className="text-lg font-bold text-gray-900 dark:text-gray-100">
                      {stage.includes('column') ? 'Data Analysis' : 
                      stage.includes('starting') || stage.includes('Starting') ? 'Counting Records' : 'File Loading'}
                    </span>
                  </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm border border-gray-100 dark:border-gray-700">
                  <div className="flex flex-col">
                    <div className="flex items-center gap-1.5 mb-1">
                      <Terminal className="h-3.5 w-3.5 text-purple-500" />
                      <span className="text-xs font-medium text-gray-700 dark:text-gray-300">System</span>
                    </div>
                    <span className="text-lg font-bold text-gray-900 dark:text-gray-100">
                      {stage.includes('Starting') || stage.includes('starting') ? 'Initializing' : 
                       stage.includes('column') ? 'Analyzing CSV' : 'Setting Up'}
                    </span>
                  </div>
                </div>
              </div>
            ) : (isProcessing && !hasError && (
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm border border-gray-100 dark:border-gray-700">
                  <div className="flex flex-col">
                    <div className="flex items-center gap-1.5 mb-1">
                      <Server className="h-3.5 w-3.5 text-indigo-500" />
                      <span className="text-xs font-medium text-gray-700 dark:text-gray-300">Records Remaining</span>
                    </div>
                    <span className="text-lg font-bold text-gray-900 dark:text-gray-100">
                      {Math.max(0, total - processedCount)}
                    </span>
                  </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm border border-gray-100 dark:border-gray-700">
                  <div className="flex flex-col">
                    <div className="flex items-center gap-1.5 mb-1">
                      <Clock className="h-3.5 w-3.5 text-purple-500" />
                      <span className="text-xs font-medium text-gray-700 dark:text-gray-300">Est. Time Left</span>
                    </div>
                    <span className="text-lg font-bold text-gray-900 dark:text-gray-100">
                      {formatTimeRemaining(timeRemaining)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
            
            {/* Processing stages */}
            {!hasError && (
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm border border-gray-100 dark:border-gray-700">
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                  Processing Stages
                </h4>
                
                <div className="space-y-2">
                  {/* File loading */}
                  <div className="flex items-center gap-2">
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center ${
                      !isLoading && (isProcessing || isComplete) 
                        ? 'bg-green-100 text-green-600' 
                        : isLoading 
                          ? 'bg-blue-100 text-blue-600' 
                          : 'bg-gray-100 text-gray-400'
                    }`}>
                      {!isLoading && (isProcessing || isComplete) ? (
                        <CheckCircle className="h-3 w-3" />
                      ) : isLoading ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <span className="text-xs">1</span>
                      )}
                    </div>
                    <span className="text-sm text-gray-700 dark:text-gray-300">File Preparation</span>
                  </div>
                  
                  {/* Records processing */}
                  <div className="flex items-center gap-2">
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center ${
                      isComplete 
                        ? 'bg-green-100 text-green-600' 
                        : isProcessing && !isComplete 
                          ? 'bg-blue-100 text-blue-600' 
                          : 'bg-gray-100 text-gray-400'
                    }`}>
                      {isComplete ? (
                        <CheckCircle className="h-3 w-3" />
                      ) : isProcessing && !isComplete ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <span className="text-xs">2</span>
                      )}
                    </div>
                    <span className="text-sm text-gray-700 dark:text-gray-300">Records Processing</span>
                  </div>
                  
                  {/* Batch information */}
                  {(isProcessing || isComplete) && batchNumber > 0 && totalBatches > 0 && (
                    <div className="text-xs text-gray-500 dark:text-gray-400 pl-7">
                      {isComplete ? (
                        `All batches completed successfully`
                      ) : (
                        `Currently on batch ${batchNumber} of ${totalBatches}`
                      )}
                    </div>
                  )}
                  
                  {/* Completion */}
                  <div className="flex items-center gap-2">
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center ${
                      isComplete 
                        ? 'bg-green-100 text-green-600' 
                        : 'bg-gray-100 text-gray-400'
                    }`}>
                      {isComplete ? (
                        <CheckCircle className="h-3 w-3" />
                      ) : (
                        <span className="text-xs">3</span>
                      )}
                    </div>
                    <span className="text-sm text-gray-700 dark:text-gray-300">Analysis Complete</span>
                  </div>
                </div>
              </div>
            )}
            
            {/* Error message */}
            {hasError && (
              <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4 border border-red-100 dark:border-red-800/30">
                <div className="flex gap-3">
                  <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
                  <div>
                    <h4 className="text-sm font-medium text-red-800 dark:text-red-300">Processing Error</h4>
                    <p className="text-sm text-red-600 dark:text-red-400 mt-1">{error || 'An error occurred during processing'}</p>
                  </div>
                </div>
                
                <div className="mt-3 text-center flex flex-col gap-2 justify-center">
                  <div className="flex justify-center space-x-2">
                    <Button
                      onClick={() => {
                        // First cancel the upload via the API
                        const sessionId = getCurrentUploadSessionId();
                        if (sessionId) {
                          cancelUpload()
                            .then(() => {
                              console.log(`Upload ${sessionId} cancelled via API`);
                              forceCloseModalMemo();
                            })
                            .catch(err => {
                              console.error('Error cancelling upload:', err);
                              // Force close anyway
                              forceCloseModalMemo();
                            });
                        } else {
                          // No sessionId, just close
                          forceCloseModalMemo();
                        }
                      }}
                      variant="destructive"
                      className="bg-red-600 hover:bg-red-700 text-white px-4"
                    >
                      Close & Cancel
                    </Button>
                  </div>
                  
                  {/* Emergency reset button */}
                  <div className="mt-2">
                    <Button
                      onClick={() => {
                        // NUCLEAR OPTION: Clear everything in localStorage
                        for (let i = 0; i < localStorage.length; i++) {
                          const key = localStorage.key(i);
                          if (key) localStorage.removeItem(key);
                        }
                        
                        // Also do a direct API cleanup
                        fetch('/api/reset-upload-sessions', {
                          method: 'POST'
                        }).then(() => {
                          console.log('EMERGENCY: Reset all upload sessions');
                          // Hard refresh the page
                          window.location.reload();
                        }).catch(e => {
                          console.error('Error in emergency reset:', e);
                          // Still reload
                          window.location.reload();
                        });
                      }}
                      variant="outline"
                      size="sm"
                      className="text-xs text-red-700 border-red-300 hover:bg-red-50"
                    >
                      Emergency Reset
                    </Button>
                  </div>
                </div>
              </div>
            )}
            
            {/* Action buttons */}
            {!isComplete && !hasError && (
              <div className="mt-3 flex flex-col sm:flex-row gap-2 justify-center">
                {/* Regular Cancel Button */}
                <Button
                  variant="destructive"
                  size="sm"
                  className="gap-1 bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700 text-white rounded-full px-5"
                  onClick={() => setShowCancelDialog(true)}
                  disabled={isCancelling}
                >
                  {isCancelling ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>Cancelling...</span>
                    </>
                  ) : (
                    <>
                      <XCircle className="h-4 w-4" />
                      <span>Cancel Upload</span>
                    </>
                  )}
                </Button>
                
                {/* Force Cancel Button - For stuck uploads */}
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-1 border border-red-300 text-red-600 hover:bg-red-50 rounded-full px-5"
                  onClick={() => handleCancel(true)}
                  disabled={isCancelling}
                >
                  {isCancelling ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>Force Cancelling...</span>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="h-4 w-4" />
                      <span>Force Cancel</span>
                    </>
                  )}
                </Button>
              </div>
            )}
            
            {/* Success message and close button */}
            {isComplete && (
              <div className="mt-3 flex justify-center">
                <Button
                  variant="default"
                  size="sm"
                  className="gap-1 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white rounded-full px-5"
                  onClick={() => forceCloseModalMemo()}
                >
                  <CheckCircle className="h-4 w-4" />
                  <span>Complete - Close</span>
                </Button>
              </div>
            )}
          </div>
        </div>
      </motion.div>
      
      {/* Cancel confirmation dialog */}
      {showCancelDialog && createPortal(
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-[10000]" onClick={() => setShowCancelDialog(false)}>
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-5 max-w-xs mx-4 shadow-xl border border-gray-200 dark:border-gray-700" 
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-start gap-3 mb-3">
              <div className="bg-red-100 dark:bg-red-900/30 p-2 rounded-full">
                <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100">Cancel Upload?</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  This will stop the current processing job. Progress will be lost and you'll need to start over.
                </p>
              </div>
            </div>
            
            <div className="flex justify-end gap-2 mt-5">
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setShowCancelDialog(false)}
                className="bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 hover:text-gray-900 border-gray-200 dark:border-gray-700 rounded-full px-4"
              >
                No, Continue
              </Button>
              <Button 
                variant="destructive"
                size="sm"
                onClick={() => handleCancel(false)} 
                className="bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700 text-white border-none rounded-full px-4"
              >
                Yes, Cancel
              </Button>
            </div>
          </motion.div>
        </div>,
        document.body
      )}
      
      {/* Animations */}
      <style>
        {`
          @keyframes gradientShift {
            0% {
              background-position: 100% 0;
            }
            100% {
              background-position: -100% 0;
            }
          }
        `}
      </style>
    </motion.div>,
    document.body
  );
}