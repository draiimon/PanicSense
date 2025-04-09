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
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { cancelUpload } from "@/lib/api";

export function UploadProgressModal() {
  const { isUploading, uploadProgress, setIsUploading, setUploadProgress } = useDisasterContext();
  const [isCancelling, setIsCancelling] = useState(false);
  const [showCancelDialog, setShowCancelDialog] = useState(false);
  
  // Check for server restart protection flag
  const serverRestartProtection = localStorage.getItem('serverRestartProtection') === 'true';
  const serverRestartTime = localStorage.getItem('serverRestartTimestamp');
  
  // CRITICAL: On component mount, check if we have state in localStorage that needs to be restored
  // This ensures the modal and its state persists through page refreshes
  useEffect(() => {
    // Always first check localStorage on component mount
    const storedSessionId = localStorage.getItem('uploadSessionId');
    const storedIsUploading = localStorage.getItem('isUploading');
    const storedProgress = localStorage.getItem('uploadProgress');
    
    // If we have stored upload state, restore it to ensure persistence
    if (storedSessionId && storedIsUploading === 'true' && storedProgress) {
      try {
        // Force modal to be visible
        if (!isUploading) {
          console.log('🚨 RESTORING UPLOAD STATE FROM LOCAL STORAGE AFTER REFRESH');
          setIsUploading(true);
          
          // Also restore the progress data
          const parsedProgress = JSON.parse(storedProgress);
          
          // Add special flag to indicate this was restored from localStorage after refresh
          parsedProgress.restoredFromLocalStorage = true;
          
          // Put the progress into context so it's visible
          setUploadProgress(parsedProgress);
          
          // Mark localStorage with a timestamp to avoid confusion with old data
          parsedProgress.restoredAt = Date.now();
          localStorage.setItem('uploadProgress', JSON.stringify(parsedProgress));
        }
      } catch (e) {
        console.error('Error restoring upload state from localStorage:', e);
      }
    }
  }, []);
  
  // Regular check with database boss - if boss says no active sessions but we're showing
  // a modal, boss is right and we should close the modal
  // BUT THIS HAS A SPECIAL EXCEPTION FOR SERVER RESTARTS
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
              
              // ALWAYS FORCE DATABASE VALUES - Critical for multi-tab consistency
              if (data.progress) {
                try {
                  // Parse database progress
                  const dbProgress = typeof data.progress === 'string' 
                    ? JSON.parse(data.progress) 
                    : data.progress;
                  
                  // Parse local progress for debugging only  
                  const localProgress = storedUploadProgress 
                    ? JSON.parse(storedUploadProgress)
                    : null;
                  
                  // STRONG CONSISTENCY: Always use database value regardless of count
                  // This is critical for multi-tab consistency - every tab must show exactly
                  // the same information. Overwrite local values EVERY time.
                  
                  // Add timestamps and mark as official database data
                  const officialData = {
                    ...dbProgress,
                    timestamp: Date.now(),
                    savedAt: Date.now(),
                    officialDbUpdate: true,
                    tabSyncTimestamp: Date.now(), // Used for tab synchronization
                    coolingDown: dbProgress.stage && dbProgress.stage.includes('pause between batches') // Track cooldown state
                  };
                  
                  // Log for debugging purposes only
                  if (localProgress && dbProgress.processed !== localProgress.processed) {
                    console.log('DATABASE COUNT AHEAD OF LOCAL - Syncing to database', 
                      `DB: ${dbProgress.processed}, Local: ${localProgress.processed}`);
                  }
                  
                  // ALWAYS save to localStorage to ensure CONSISTENT display across tabs
                  localStorage.setItem('uploadProgress', JSON.stringify(officialData));
                  
                  // Create special event for cross-tab communication
                  try {
                    // Use localStorage event for cross-tab communication
                    localStorage.setItem('lastTabSync', JSON.stringify({
                      timestamp: Date.now(),
                      processed: dbProgress.processed,
                      stage: dbProgress.stage
                    }));
                  } catch (e) {
                    // Ignore errors in cross-tab communication
                  }
                } catch (e) {
                  console.error('Error comparing progress data:', e);
                }
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
  
  // Manually close the modal and clean up both localStorage and database
  // This improved version handles race conditions and prevents UI flicker
  const forceCloseModal = async () => {
    // Set local cancelling state immediately to prevent multiple calls
    setIsCancelling(true);
    
    try {
      // First try to cancel the upload through the API to ensure database cleanup
      // This ensures the server-side cleanup occurs, deleting the session from the database
      const sessionId = localStorage.getItem('uploadSessionId');
      if (sessionId) {
        // Try to cancel through the API first (which will delete the session from the database)
        await cancelUpload();
        console.log('Force close: Upload cancelled through API to ensure database cleanup');
      }
    } catch (error) {
      console.error('Error cancelling upload during force close:', error);
    } finally {
      // Always clean up localStorage regardless of API success/failure
      // This prevents stale state that could cause UI flickering
      localStorage.removeItem('isUploading');
      localStorage.removeItem('uploadProgress');
      localStorage.removeItem('uploadSessionId');
      localStorage.removeItem('lastProgressTimestamp');
      localStorage.removeItem('lastDatabaseCheck');
      
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
    }
  };
  
  // Handle cancel button click
  const handleCancel = async () => {
    if (isCancelling) return;
    
    setShowCancelDialog(false);
    setIsCancelling(true);
    try {
      const result = await cancelUpload();
      
      if (result.success) {
        // Force close the modal instead of waiting for events
        forceCloseModal();
      } else {
        setIsCancelling(false);
      }
    } catch (error) {
      console.error('Error cancelling upload:', error);
      // Even on error, force close the modal
      forceCloseModal();
    }
  };

  // Don't render the modal if not uploading
  if (!isUploading) return null;

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
    error = ''
  } = uploadProgress;
  
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
  
  // IMPROVED COOLDOWN DETECTION: Check for "pause between batches" with enhanced detection
  // This ensures we properly display the cooldown state especially after page refresh
  const isPaused = stageLower.includes('pause between batches') || 
                  stageLower.includes('cooldown') || 
                  (uploadProgress.coolingDown === true);
                  
  // Add a localStorage entry to improve persistence of cooldown state
  if (isPaused && !localStorage.getItem('cooldownActive')) {
    localStorage.setItem('cooldownActive', 'true');
    localStorage.setItem('cooldownStartedAt', Date.now().toString());
  } else if (!isPaused && localStorage.getItem('cooldownActive')) {
    localStorage.removeItem('cooldownActive');
    localStorage.removeItem('cooldownStartedAt');
  }
  
  const isLoading = stageLower.includes('loading') || stageLower.includes('preparing');
  const isProcessingRecord = stageLower.includes('processing record') || stageLower.includes('completed record');
  
  // Consider any active work state as "processing"
  const isProcessing = isProcessingRecord || isPaused || stageLower.includes('processing');
  
  // Only set complete when explicitly mentioned OR when we've processed everything
  // Require 99% completion to avoid premature "Analysis Complete!"
  const isReallyComplete = stageLower.includes('completed all') || 
                        stageLower.includes('analysis complete') || 
                        (rawProcessed >= total * 0.99 && total > 100);
  
  // Final completion state
  const isComplete = isReallyComplete;
  
  // Calculate completion percentage safely - ensure it's visible when processing
  const percentComplete = total > 0 
    ? Math.min(100, Math.max(isProcessing ? 1 : 0, Math.round((processedCount / total) * 100)))
    : 0;
  
  // Check for cancellation
  const isCancelled = stageLower.includes('cancel');
  
  // Improved error detection with auto-close
  const hasError = stageLower.includes('error');
  
  // Auto-close on error or completion after a delay
  useEffect(() => {
    if (hasError || isComplete || isCancelled) {
      // Start a timer to auto-close after 5 seconds
      const autoCloseTimer = setTimeout(() => {
        console.log(`🔄 Auto-closing upload modal: Error=${hasError}, Complete=${isComplete}, Cancelled=${isCancelled}`);
        
        // Clear all localStorage state
        localStorage.removeItem('isUploading');
        localStorage.removeItem('uploadProgress');
        localStorage.removeItem('uploadSessionId');
        localStorage.removeItem('lastProgressTimestamp');
        localStorage.removeItem('lastDatabaseCheck');
        localStorage.removeItem('serverRestartProtection');
        localStorage.removeItem('serverRestartTimestamp');
        localStorage.removeItem('cooldownActive');
        localStorage.removeItem('cooldownStartedAt');
        localStorage.removeItem('lastTabSync');
        
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
        
        // Update context state to close the modal
        setIsUploading(false);
      }, 5000); // 5 second delay for user to see the final status
      
      // Cleanup timer on unmount
      return () => clearTimeout(autoCloseTimer);
    }
  }, [hasError, isComplete, isCancelled, setIsUploading]);
  
  // SUPER ENHANCED 2025 VERSION: Calculate time remaining in human-readable format
  // with critical improvements to prevent time jumps and ensure consistent estimates
  const formatTimeRemaining = (seconds: number): string => {
    // NEVER SHOW "CALCULATING..." - Always show a time estimate
    // Even if no time is provided or if it's 0, calculate one based on record count and processing speed

    // Detect cooldown state from stage text or coolingDown flag
    const isCoolingDown = stageLower.includes('pause between batches') || 
                          stageLower.includes('cooling down') || 
                          uploadProgress.coolingDown === true;
    
    // Constants specified by user requirements for consistency across all calculations
    const SECONDS_PER_RECORD = 3; // Each record takes exactly 3 seconds to process (user requirement)
    const BATCH_SIZE = 30; // Process in batches of 30 records (user requirement)
    const COOLDOWN_SECONDS = 60; // Exactly 60 second cooldown between batches (user requirement)
    
    // For datasets over 3000 records, add 50% more processing time to prevent sudden time jumps
    // This adjustment should be applied consistently at all calculation points
    const adjustmentFactor = total > 3000 ? 1.5 : 1.0;
    
    // Calculate remaining records and batches with enhanced precision
    const recordsRemaining = Math.max(0, total - processedCount);
    const currentBatch = Math.ceil(processedCount / BATCH_SIZE);
    const totalBatches = Math.ceil(total / BATCH_SIZE);
    const remainingBatches = Math.max(0, totalBatches - currentBatch);
    
    // Enhanced calculation of processing and cooldown times
    const processingTimeSeconds = recordsRemaining * SECONDS_PER_RECORD * adjustmentFactor;
    const cooldownTimeSeconds = remainingBatches * COOLDOWN_SECONDS;
    
    // If we're currently in a cooldown period, extract the remaining cooldown time
    let currentCooldownRemaining = 0;
    if (isCoolingDown) {
      // Extract remaining seconds from stage text (e.g., "60-second pause between batches: 42 seconds remaining")
      const cooldownMatch = stageLower.match(/(\d+)\s*seconds? remaining/i);
      if (cooldownMatch && cooldownMatch[1]) {
        currentCooldownRemaining = parseInt(cooldownMatch[1], 10);
        console.log(`📊 Cooldown detected: ${currentCooldownRemaining} seconds remaining`);
      }
    }
    
    // Combine for total estimated time (current cooldown + remaining processing + remaining cooldowns)
    let calculatedTimeRemaining = currentCooldownRemaining + processingTimeSeconds + cooldownTimeSeconds;
    
    // Force minimum time thresholds based on total records and current progress
    // This ensures larger datasets always show realistic times for the UI
    let minimumTimeSeconds = 0;
    
    // Enhanced minimum threshold logic with more granular stages for larger datasets
    if (total > 3000) {
      // For datasets over 3000 records (enhanced thresholds to prevent time jumps)
      if (processedCount < total * 0.05) {
        // Very beginning of processing (first 5%)
        minimumTimeSeconds = 5 * 60 * 60; // 5 hours minimum for initial display
      } else if (processedCount < total * 0.1) {
        // Early in processing (5-10%)
        minimumTimeSeconds = 4 * 60 * 60; // 4 hours minimum
      } else if (processedCount < total * 0.25) {
        // First quarter (10-25%)
        minimumTimeSeconds = 3 * 60 * 60; // 3 hours minimum
      } else if (processedCount < total * 0.5) {
        // Approaching middle (25-50%)
        minimumTimeSeconds = 2 * 60 * 60; // 2 hours minimum
      } else if (processedCount < total * 0.75) {
        // Third quarter (50-75%)
        minimumTimeSeconds = 1 * 60 * 60; // 1 hour minimum
      } else if (processedCount < total * 0.9) {
        // Final stretch (75-90%)
        minimumTimeSeconds = 30 * 60; // 30 minutes minimum
      } else {
        // Almost done (90-100%)
        minimumTimeSeconds = 15 * 60; // 15 minutes minimum
      }
    } else if (total > 1000) {
      // For medium-sized datasets (1000-3000 records)
      if (processedCount < total * 0.25) {
        minimumTimeSeconds = 90 * 60; // 1.5 hour for early stages
      } else if (processedCount < total * 0.5) {
        minimumTimeSeconds = 60 * 60; // 1 hour for middle stages
      } else if (processedCount < total * 0.75) {
        minimumTimeSeconds = 30 * 60; // 30 minutes for later stages
      }
    }
    
    // Apply minimum time if our calculation is too optimistic
    if (calculatedTimeRemaining < minimumTimeSeconds) {
      calculatedTimeRemaining = minimumTimeSeconds;
    }
    
    // CRITICAL: Prevent sudden time jumps (like 2.5 hours to 5 minutes) by smoothing changes
    // This is specifically to fix the user-reported issue with larger datasets
    if (timeRemaining > 0) {
      // For larger time estimates, use more aggressive smoothing to prevent drastic drops
      // Adjust the max reduction percentage based on current progress
      let maxReductionPercentage = 0.05; // Default: Maximum 5% reduction per update
      
      // Apply more conservative smoothing for larger datasets and longer timeframes
      if (total > 3000 && timeRemaining > 60 * 60) {
        maxReductionPercentage = 0.03; // Only 3% reduction for very large datasets with hours remaining
      } else if (timeRemaining > 30 * 60) {
        maxReductionPercentage = 0.04; // 4% for medium timeframes
      }
      
      const minAllowedEstimate = timeRemaining * (1 - maxReductionPercentage);
      
      // Only allow small percentage drop at a time for large time estimates
      if (timeRemaining > 15 * 60 && calculatedTimeRemaining < minAllowedEstimate) {
        calculatedTimeRemaining = minAllowedEstimate;
        console.log('🔒 PREVENTING TIME JUMP: Smoothed time estimate', 
          `Old: ${Math.round(timeRemaining/60)}m, New: ${Math.round(calculatedTimeRemaining/60)}m`);
      }
    }
    
    // Finally, if we have a server-provided time estimate that's reasonable, use weighted average
    if (seconds > 0 && seconds !== calculatedTimeRemaining) {
      // For non-zero server times, use a weighted average to smooth transitions
      // Give more weight to our calculated time to ensure consistency
      calculatedTimeRemaining = (calculatedTimeRemaining * 0.75) + (seconds * 0.25);
    }
    
    // Always return a formatted time string, NEVER "calculating..."
    // Set absolute minimum to 30 seconds to avoid showing unrealistically short times
    const actualSeconds = Math.max(30, calculatedTimeRemaining);
    
    // Format the time in a user-friendly way
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
                      System is paused between batches to prevent overloading
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
                      {/* Enhanced time calculation - will NEVER show "calculating..." 
                          Always calculates a realistic estimate based on records & processing speed */}
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
                
                <div className="mt-3 text-center">
                  <Button
                    onClick={() => forceCloseModal()}
                    variant="destructive"
                    className="bg-red-600 hover:bg-red-700 text-white px-4"
                  >
                    Close
                  </Button>
                </div>
              </div>
            )}
            
            {/* Action buttons */}
            {!isComplete && !hasError && (
              <div className="mt-3 flex justify-center">
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
              </div>
            )}
            
            {/* Success message and close button */}
            {isComplete && (
              <div className="mt-3 flex justify-center">
                <Button
                  variant="default"
                  size="sm"
                  className="gap-1 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white rounded-full px-5"
                  onClick={() => forceCloseModal()}
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
                onClick={handleCancel}
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