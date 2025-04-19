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
  Terminal,
  Trash2
} from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { createPortal } from "react-dom";
import React, { useEffect, useState, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { ToastAction } from "@/components/ui/toast";
import { 
  createBroadcastListener, 
  cleanupUploadState, 
  markUploadCompleted, 
  isUploadCompleted,
  broadcastMessage
} from "@/lib/synchronization-manager";
import { cancelUpload, getCurrentUploadSessionId } from "@/lib/api";
import { nuclearCleanup, listenForNuclearCleanup } from "@/hooks/use-nuclear-cleanup";

// Add a global TypeScript interface for window
declare global {
  interface Window {
    _activeEventSources?: Record<string, EventSource>;
  }
}

export function UploadProgressModal() {
  const { isUploading, uploadProgress, setIsUploading, setUploadProgress } = useDisasterContext();
  const [isCancelling, setIsCancelling] = useState(false);
  const [showCancelDialog, setShowCancelDialog] = useState(false);
  const { toast } = useToast(); // Initialize toast hook
  
  // Extract values from uploadProgress at the top level
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
  } = uploadProgress || {};
  
  // Set up listener for nuclear cleanup broadcasts from other tabs
  useEffect(() => {
    const removeNuclearListener = listenForNuclearCleanup();
    return () => removeNuclearListener();
  }, []);
  
  // Define a clean-up function that will be used for both direct modal closing and broadcast handling
  const cleanupAndClose = React.useCallback(() => {
    console.log('🧹 Cleanup and close function triggered');
    
    // Try the nuclear cleanup option first to ensure complete cleanup
    nuclearCleanup();
    
    // Fallback to regular cleanup if nuclear fails for any reason
    cleanupUploadState();
    
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
    
    console.log('🧹 MODAL CLOSED - ALL STATE CLEARED');
  }, [setIsUploading, setIsCancelling]);
  
  // Set up broadcast listener using the new synchronization manager
  useEffect(() => {
    // Before setting up listeners, check if another tab already marked completion
    if (isUploadCompleted() && isUploading) {
      console.log('🔄 Another tab already completed the upload, closing this modal');
      // Use a short delay to allow UI to update first
      setTimeout(() => {
        cleanupAndClose();
      }, 500);
    }
    
    // Create a listener for broadcast messages using our synchronization manager
    const removeListener = createBroadcastListener({
      onUploadProgress: (progress) => {
        console.log('📊 Received progress update from another tab');
        setUploadProgress(progress);
      },
      
      onUploadComplete: (progress) => {
        console.log('🏁 Received completion notification from another tab');
        
        // Force this tab to show the upload modal if we're not already showing it
        if (!isUploading) {
          console.log('🔄 This tab was not showing the upload modal, forcing it to show');
          setIsUploading(true);
        }
        
        // Update progress state to show completion
        setUploadProgress({
          ...progress,
          stage: 'Analysis complete',
          processed: progress.total || 100,
          total: progress.total || 100,
          currentSpeed: 0,
          timeRemaining: 0
        });
        
        // Set a timer to auto-close
        const completionDelay = 3000; // 3 seconds
        console.log(`📊 Will auto-close after ${completionDelay}ms due to completion message`);
        
        setTimeout(() => {
          console.log('⏰ AUTO-CLOSE TRIGGERED BY COMPLETION MESSAGE');
          cleanupAndClose();
        }, completionDelay);
      },
      
      onUploadCleanup: () => {
        console.log('🧹 Received cleanup message from another tab');
        if (isUploading) {
          console.log('Closing this modal due to cleanup message from another tab');
          cleanupAndClose();
        }
      },
      
      onUploadCancelled: () => {
        console.log('🔥 Received cancellation message from another tab');
        cleanupAndClose();
      }
    });
    
    // Return cleanup function
    return removeListener;
  }, [setUploadProgress, cleanupAndClose, isUploading, setIsUploading, uploadProgress]);
  
  // Check for server restart protection flag
  // Only consider server restart protection if explicitly set
  // Check for both server restart AND completion status - never show server restart mode for completed uploads
  const serverRestartDetected = localStorage.getItem('serverRestartProtection') === 'true';
  const serverRestartTime = localStorage.getItem('serverRestartTimestamp');
  const uploadCompleted = isUploadCompleted();
  
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
    
    // Anti-flicker prevention: if we already know it's completed, don't check again
    if (isUploadCompleted()) {
      console.log('🏁 Upload already marked as completed, skipping API check');
      return;
    }
    
    // Using stateful vars for debounce without requiring state updates
    const state = {
      lastCheckTime: 0,
      completionVerified: false
    };
    
    // Much more robust completion check with debounce
    const checkCompletionStatus = async () => {
      const now = Date.now();
      const POLL_THROTTLE = 3000; // 3 seconds between polls
      
      // Don't check too frequently to avoid overwhelming the server
      if (now - state.lastCheckTime < POLL_THROTTLE) {
        return;
      }
      
      // Also don't check if we've already verified completion
      if (state.completionVerified || isUploadCompleted()) {
        return;
      }
      
      // Update our check timestamp
      state.lastCheckTime = now;
      
      try {
        const response = await fetch('/api/upload-complete-check');
        if (!response.ok) return;
        
        const data = await response.json();
        
        // If the server says upload is complete, mark it completed
        if (data.uploadComplete && data.sessionId) {
          console.log('🌟 SERVER CONFIRMS UPLOAD IS COMPLETE!', data.sessionId);
          
          // Mark this instance as having verified completion
          state.completionVerified = true;
          
          // Update our local state first
          const finalProgress = {
            ...uploadProgress,
            stage: 'Analysis complete',
            processed: uploadProgress.total || 100,
            total: uploadProgress.total || 100,
            currentSpeed: 0,
            timeRemaining: 0
          };
          
          setUploadProgress(finalProgress);
          
          // Use the synchronization manager to mark it complete and notify other tabs
          markUploadCompleted(finalProgress);
          
          // Auto-close after a delay
          setTimeout(() => {
            console.log('⏰ AUTO-CLOSE TRIGGERED BY COMPLETION VERIFICATION');
            cleanupAndClose();
          }, 3000);
        }
      } catch (error) {
        console.error('Error checking upload completion status:', error);
      }
    };
    
    // Initial delay before first check to let the UI stabilize
    const initialCheckTimeout = setTimeout(checkCompletionStatus, 1000);
    
    // Check periodically but not too frequently
    const intervalId = setInterval(checkCompletionStatus, 3000);
    
    return () => {
      clearTimeout(initialCheckTimeout);
      clearInterval(intervalId);
    };
  }, [isUploading, uploadProgress, setUploadProgress, cleanupAndClose]);
  
  // Add auto-close timer for both "Analysis complete" and error states
  // Memoize forceCloseModal to prevent unnecessary re-renders
  const forceCloseModalMemo = React.useCallback(async () => {
    // Set local cancelling state immediately to prevent multiple calls
    setIsCancelling(true);
    
    try {
      // First try to cancel the upload through the API to ensure database cleanup
      // This ensures the server-side cleanup occurs, deleting the session from the database
      const sessionId = localStorage.getItem('uploadSessionId');
      
      // Use the synchronization manager to clean up all state
      cleanupUploadState();
      
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
      broadcastMessage('upload_force_cancelled');
      
      console.log('🧹 MODAL FORCED CLOSED - ALL STATE CLEARED');
    }
  }, [setIsUploading, setIsCancelling]);

  // Effect for handling "Analysis complete" and error states
  useEffect(() => {
    // INSTANT CLOSE FOR ERRORS, brief delay for completion
    if (isUploading) {
      if (stage === 'Upload Error') {
        // INSTANT CLEANUP FOR ERRORS - close immediately with tiny delay
        console.log(`🚨 ERROR DETECTED - CLOSING IMMEDIATELY`);
        
        // Use minimal delay (50ms) just to ensure state can be seen
        const closeTimerId = setTimeout(() => {
          console.log(`⏰ ERROR AUTO-CLOSE TRIGGERED IMMEDIATELY`);
          
          // Use synchronization manager for cleanup
          cleanupUploadState();
          
          forceCloseModalMemo(); // Close the modal automatically
        }, 50); // Super short delay
        
        return () => clearTimeout(closeTimerId);
      }
      else if (stage === 'Analysis complete') {
        // For completion, show success briefly (3 seconds)
        const completionDelay = autoCloseDelay || 3000; // Default to 3 seconds
        console.log(`🎯 Analysis complete - will auto-close after ${completionDelay}ms`);
        
        // Create final progress object
        const finalProgress = {
          ...uploadProgress,
          stage: 'Analysis complete',
          processed: total,
          total: total,
          currentSpeed: 0,
          timeRemaining: 0
        };
        
        // Use the synchronization manager to mark as complete and broadcast to other tabs
        markUploadCompleted(finalProgress);
        
        // Set a timer to auto-close this tab
        const closeTimerId = setTimeout(() => {
          console.log(`⏰ COMPLETION AUTO-CLOSE TRIGGERED AT ${completionDelay}ms`);
          cleanupAndClose(); // Use the standard cleanup method
        }, completionDelay);
        
        return () => clearTimeout(closeTimerId);
      }
    }
  }, [isUploading, stage, total, uploadProgress, forceCloseModalMemo, cleanupAndClose, autoCloseDelay]);

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
  
  // Calculate time remaining in human-readable format with improved batch-aware logic
  const formatTimeRemaining = (seconds: number): string => {
    if (!seconds || seconds <= 0) return 'calculating...';
    
    // Calculate a realistic time remaining based on processed records, speed, and batch cooldowns
    // This ensures we're showing updated time even if server doesn't update timeRemaining
    let calculatedTimeRemaining = seconds;
    
    // If we're in a batch pause state, adjust calculation method
    if (isPaused && cooldownTime !== null) {
      // Use the cooldown time directly if available
      return `${cooldownTime} sec cooldown`;
    } else if (currentSpeed > 0 && total > processedCount) {
      // For normal processing, calculate based on current speed and records left
      const recordsRemaining = total - processedCount;
      const processingTime = recordsRemaining / currentSpeed;
      
      // Add batch cooldown estimation - for every 30 records, add 60 seconds of cooldown
      // Only if we have multiple batches
      let cooldownEstimate = 0;
      if (totalBatches > 1) {
        const remainingBatches = Math.ceil(recordsRemaining / 30);
        if (remainingBatches > 0) {
          cooldownEstimate = (remainingBatches - 1) * 60; // 60 sec cooldown between batches
        }
      }
      
      // Combine processing time and cooldown time
      calculatedTimeRemaining = processingTime + cooldownEstimate;
    }
    
    // Use the smaller of the two values but only if server-provided time is reasonable
    // If server time is much lower, it might be missing cooldown estimates
    const actualSeconds = seconds > 10 ? Math.min(calculatedTimeRemaining, seconds) : calculatedTimeRemaining;
    
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
      {/* Simple backdrop */}
      <div className="absolute inset-0 bg-black/60"></div>

      {/* Content Container */}
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        transition={{ duration: 0.3, type: "spring", stiffness: 300, damping: 30 }}
        className="relative bg-white dark:bg-gray-900 rounded-xl overflow-hidden w-full max-w-md mx-4 shadow-xl border border-gray-200 dark:border-gray-800"
      >
        {/* Header */}
        <div className="bg-blue-600 p-4 text-white">
          {/* Simple header content */}
          
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
              </div>
            ) : isComplete ? (
              <div className="py-6 flex flex-col items-center justify-center">
                <CheckCircle className="h-14 w-14 text-white mb-3" />
                <span className="text-lg font-medium text-white">Analysis Successfully Completed</span>
                <span className="text-sm text-white/70 mt-1">You can now view the results</span>
              </div>
            ) : hasError ? (
              <div className="py-5 flex flex-col items-center justify-center">
                <XCircle className="h-12 w-12 text-white mb-3" />
                <span className="text-lg font-medium text-white">Processing Error</span>
                <span className="text-sm text-white/70 mt-1">{error || "An unexpected error occurred"}</span>
              </div>
            ) : serverRestartProtection ? (
              <div className="py-5 flex flex-col items-center justify-center">
                <Server className="h-12 w-12 text-white mb-3" />
                <span className="text-lg font-medium text-white">Server Restarted</span>
                <span className="text-sm text-white/70 mt-1">Restoring your upload...</span>
              </div>
            ) : (
              <div className="flex items-center">
                <div className="text-center mr-4">
                  <div className="text-3xl font-bold">{percentComplete}%</div>
                  <div className="text-sm text-white/80">Complete</div>
                </div>
                <div className="h-14 w-[1px] bg-white/20 mx-2"></div>
                <div className="text-center ml-4">
                  <div className="text-3xl font-bold">{processedCount}</div>
                  <div className="text-sm text-white/80">of {total} Records</div>
                </div>
              </div>
            )}
          </motion.div>
        </div>
        
        {/* Progress Content */}
        <div className="p-5">
          {/* Progress bar */}
          {!isComplete && !hasError && (
            <div className="mb-4">
              <div className="h-3 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                <motion.div 
                  className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${percentComplete}%` }}
                  transition={{ duration: 0.5 }}
                  style={{
                    backgroundSize: '200% 100%',
                    animation: 'gradientShift 2s linear infinite'
                  }}
                ></motion.div>
              </div>
            </div>
          )}
          
          {/* Status information */}
          <div className="space-y-3">
            {/* Current stage */}
            <div className="flex items-start">
              <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center mt-0.5 flex-shrink-0">
                <FileText className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              </div>
              <div className="ml-3">
                <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400">Current Stage</h4>
                <p className="text-base font-medium text-gray-900 dark:text-gray-200 mt-0.5">{stage}</p>
              </div>
            </div>
            
            {/* Server stats - only show if processing */}
            {isProcessing && !isComplete && !hasError && (
              <>
                {/* Processing speed */}
                <div className="flex items-start">
                  <div className="w-8 h-8 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center mt-0.5 flex-shrink-0">
                    <BarChart3 className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                  </div>
                  <div className="ml-3">
                    <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400">Processing Speed</h4>
                    <p className="text-base font-medium text-gray-900 dark:text-gray-200 mt-0.5">
                      {currentSpeed > 0 ? `${currentSpeed.toFixed(2)} records/sec` : 'Calculating...'}
                    </p>
                  </div>
                </div>
                
                {/* Time remaining */}
                <div className="flex items-start">
                  <div className="w-8 h-8 rounded-full bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center mt-0.5 flex-shrink-0">
                    <Clock className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
                  </div>
                  <div className="ml-3">
                    <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400">Estimated Time</h4>
                    <p className="text-base font-medium text-gray-900 dark:text-gray-200 mt-0.5">
                      {isPaused ? (
                        cooldownTime !== null ? 
                          `Resume in ${cooldownTime} seconds` : 
                          'Paused between batches'
                      ) : (
                        formatTimeRemaining(timeRemaining)
                      )}
                    </p>
                  </div>
                </div>
                
                {/* Batch information - only show if we have multiple batches */}
                {totalBatches > 1 && batchNumber > 0 && (
                  <div className="flex items-start">
                    <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center mt-0.5 flex-shrink-0">
                      <Database className="h-4 w-4 text-green-600 dark:text-green-400" />
                    </div>
                    <div className="ml-3">
                      <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400">Batch Progress</h4>
                      <p className="text-base font-medium text-gray-900 dark:text-gray-200 mt-0.5">
                        Batch {batchNumber} of {totalBatches} 
                        {batchProgress ? ` (${Math.round(batchProgress * 100)}%)` : ''}
                      </p>
                    </div>
                  </div>
                )}
              </>
            )}
            
            {/* Stats summary - show for completion */}
            {isComplete && (
              <div className="flex items-start">
                <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center mt-0.5 flex-shrink-0">
                  <BarChart3 className="h-4 w-4 text-green-600 dark:text-green-400" />
                </div>
                <div className="ml-3">
                  <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400">Processing Summary</h4>
                  <p className="text-base font-medium text-gray-900 dark:text-gray-200 mt-0.5">
                    {processingStats.successCount} Records Analyzed
                    {processingStats.errorCount > 0 && `, ${processingStats.errorCount} Errors`}
                  </p>
                </div>
              </div>
            )}
            
            {/* Error details */}
            {hasError && error && (
              <div className="flex items-start">
                <div className="w-8 h-8 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center mt-0.5 flex-shrink-0">
                  <Terminal className="h-4 w-4 text-red-600 dark:text-red-400" />
                </div>
                <div className="ml-3">
                  <h4 className="text-sm font-medium text-gray-600 dark:text-gray-400">Error Details</h4>
                  <p className="text-sm text-red-600 dark:text-red-400 mt-0.5">
                    {error}
                  </p>
                  
                  {/* Close & cancel button for errors */}
                  <div className="mt-3">
                    <Button
                      size="sm"
                      onClick={() => {
                        const sessionId = localStorage.getItem('uploadSessionId');
                        
                        // Try to ensure server-side cleanup too if possible
                        if (sessionId) {
                          cancelUpload(true)
                            .then(() => {
                              console.log('Cancelled upload due to error');
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
                  <div className="mt-2 flex space-x-2">
                    <Button
                      onClick={() => {
                        // Clean up all state
                        cleanupUploadState();
                        
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
                    
                    <Button
                      onClick={() => {
                        toast({
                          title: "Nuclear Cleanup Initiated",
                          description: "Performing complete cleanup across all browser tabs",
                          variant: "destructive",
                        });
                        
                        // Perform nuclear cleanup
                        nuclearCleanup();
                        
                        // Force close the modal
                        forceCloseModalMemo();
                        
                        // Also do server-side cleanup
                        fetch('/api/reset-upload-sessions', {
                          method: 'POST'
                        }).then(() => {
                          console.log('☢️ NUCLEAR: Reset all upload sessions');
                          // Hard refresh the page after a short delay
                          setTimeout(() => {
                            window.location.reload();
                          }, 1000);
                        }).catch(e => {
                          console.error('Error in nuclear reset:', e);
                          // Still reload after a delay
                          setTimeout(() => {
                            window.location.reload();
                          }, 1000);
                        });
                      }}
                      variant="outline"
                      size="sm"
                      className="text-xs bg-red-800 text-white border-red-900 hover:bg-red-900 hover:text-white flex items-center gap-1"
                    >
                      <Trash2 className="h-3 w-3" />
                      <span>Nuclear Cleanup</span>
                    </Button>
                  </div>
                </div>
              </div>
            )}
            
            {/* Action buttons */}
            {!isComplete && !hasError && (
              <div className="mt-3 space-y-2">
                <div className="flex flex-col sm:flex-row gap-2 justify-center">
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
                
                {/* Nuclear Reset Button */}
                <div className="flex justify-center">
                  <Button
                    onClick={() => {
                      // Perform nuclear cleanup
                      nuclearCleanup();

                      toast({
                        title: "Nuclear Cleanup",
                        description: "Performing complete cleanup across all tabs",
                        variant: "destructive",
                      });
                      
                      // Also do a direct API cleanup
                      fetch('/api/reset-upload-sessions', {
                        method: 'POST'
                      }).catch(e => {
                        console.error('Error in nuclear reset API call:', e);
                      });
                      
                      // Force close the modal immediately
                      setIsUploading(false);
                      setIsCancelling(false);
                      
                      // Hard refresh after short delay
                      setTimeout(() => {
                        window.location.reload();
                      }, 500);
                    }}
                    variant="outline"
                    size="sm"
                    className="flex items-center gap-1 text-xs mt-1 bg-gray-800 text-white hover:bg-gray-900 border-none px-3 py-1 rounded"
                  >
                    <Trash2 className="h-3 w-3" />
                    <span>Nuclear Reset</span>
                  </Button>
                </div>
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