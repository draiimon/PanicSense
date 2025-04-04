import { motion, AnimatePresence } from "framer-motion";
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
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { cancelUpload } from "@/lib/api";

// Anti-flicker mechanism - add debounce to modal visibility changes
const DEBOUNCE_DELAY = 300; // milliseconds to wait before hiding the modal

export function UploadProgressModal() {
  const { isUploading, uploadProgress, setIsUploading } = useDisasterContext();
  const [isCancelling, setIsCancelling] = useState(false);
  const [showCancelDialog, setShowCancelDialog] = useState(false);
  const [visibleModal, setVisibleModal] = useState(false);
  
  // Use refs to track timeouts to prevent memory leaks
  const hideTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Use a ref to track operations in progress
  const operationInProgressRef = useRef(false);
  
  // Extract uploadProgress values for display
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
  
  // SIMPLIFIED STAGE DETECTION LOGIC
  // Convert stage to lowercase once for all checks
  const stageLower = stage.toLowerCase();
  
  // Check if we're in the initializing phase
  // Include the initial loading state when application is started or refreshed
  // and restoring an in-progress upload
  const isInitializing = rawProcessed === 0 || 
                        stageLower.includes('initializing') || 
                        stageLower.includes('loading csv file') ||
                        stageLower.includes('file loaded') ||
                        stageLower.includes('identifying columns') || 
                        stageLower.includes('identified data columns') ||
                        stageLower.includes('preparing');
  
  // Keep original server values for display
  const processedCount = rawProcessed;
  
  // Basic state detection - clear, explicit flags
  const isPaused = stageLower.includes('pause between batches');
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
  
  // Improved error detection
  const hasError = stageLower.includes('error');
  
  // Handle modal visibility with anti-flicker protection
  useEffect(() => {
    if (isUploading) {
      // Clear any pending hide operations when showing the modal
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current);
        hideTimeoutRef.current = null;
      }
      setVisibleModal(true);
    } else {
      // Add a delay before hiding to prevent flicker from quick state changes
      hideTimeoutRef.current = setTimeout(() => {
        setVisibleModal(false);
        hideTimeoutRef.current = null;
      }, DEBOUNCE_DELAY);
    }
    
    // Cleanup on unmount
    return () => {
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current);
      }
    };
  }, [isUploading]);
  
  // Calculate time remaining in human-readable format
  const formatTimeRemaining = (seconds: number): string => {
    if (!seconds || seconds <= 0) return 'calculating...';
    
    // Less than a minute
    if (seconds < 60) return `${Math.ceil(seconds)} sec`;
    
    // Calculate days, hours, minutes, seconds
    const days = Math.floor(seconds / 86400); // 86400 seconds in a day
    const hours = Math.floor((seconds % 86400) / 3600); // 3600 seconds in an hour
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = Math.ceil(seconds % 60);
    
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
  
  // Safely close the modal and clean up all resources
  const forceCloseModal = useCallback(async () => {
    // Prevent multiple concurrent operations
    if (operationInProgressRef.current) return;
    operationInProgressRef.current = true;
    
    // Set local cancelling state immediately
    setIsCancelling(true);
    
    try {
      // First try to cancel the upload through the API to ensure database cleanup
      const sessionId = localStorage.getItem('uploadSessionId');
      if (sessionId) {
        try {
          await cancelUpload();
          console.log('Force close: Upload cancelled through API to ensure database cleanup');
        } catch (apiError) {
          console.error('Error cancelling upload via API:', apiError);
        }
      }
    } finally {
      // Always perform thorough cleanup
      
      // 1. Clean up localStorage
      localStorage.removeItem('isUploading');
      localStorage.removeItem('uploadProgress');
      localStorage.removeItem('uploadSessionId');
      localStorage.removeItem('lastProgressTimestamp');
      localStorage.removeItem('lastDatabaseCheck');
      localStorage.removeItem('uploadStartTime');
      
      // 2. Close all EventSource connections
      if (window._activeEventSources) {
        Object.values(window._activeEventSources).forEach(source => {
          try {
            source.close();
          } catch (e) {
            // Ignore errors on close
          }
        });
        window._activeEventSources = {};
      }
      
      // 3. Update context state AFTER cleanup is complete
      setIsUploading(false);
      setIsCancelling(false);
      
      // 4. Release operation lock
      operationInProgressRef.current = false;
    }
  }, [setIsUploading]);
  
  // Handle cancel button click with improved error handling
  const handleCancel = useCallback(async () => {
    if (isCancelling || operationInProgressRef.current) return;
    
    setShowCancelDialog(false);
    setIsCancelling(true);
    operationInProgressRef.current = true;
    
    try {
      const result = await cancelUpload();
      
      if (result.success) {
        forceCloseModal();
      } else {
        // Even on API failure, force close the modal
        forceCloseModal();
      }
    } catch (error) {
      console.error('Error cancelling upload:', error);
      // Always force close on any error
      forceCloseModal();
    }
  }, [isCancelling, forceCloseModal]);

  return (
    <AnimatePresence>
      {visibleModal && createPortal(
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
                    <span className="text-lg font-medium text-white">Preparing Upload...</span>
                    <span className="text-sm text-white/70 mt-1">Initializing processing service</span>
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
                      {hasError ? (
                        <AlertCircle className="h-5 w-5" />
                      ) : isComplete ? (
                        <CheckCircle className="h-5 w-5" />
                      ) : (
                        <Loader2 className="h-5 w-5 animate-spin" />
                      )}
                    </div>
                    <div className="flex-1">
                      <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100">Current Status</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-300 mt-0.5">
                        {stage}
                      </p>
                      {isPaused && (
                        <p className="text-xs text-amber-600 mt-1 font-medium">
                          System is paused between batches to prevent overloading
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
                          Preparing
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
                          Starting
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
                
                {/* Action buttons */}
                <div className="mt-5 flex items-center justify-between">
                  {/* Only show cancel when not complete and not in the process of cancelling */}
                  {!isComplete && !isCancelling && !hasError && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-1"
                      onClick={() => setShowCancelDialog(true)}
                    >
                      <XCircle className="h-4 w-4" />
                      <span>Cancel</span>
                    </Button>
                  )}
                  
                  {/* Show cancelling indicator */}
                  {isCancelling && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-1 opacity-70 cursor-not-allowed"
                      disabled
                    >
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>Cancelling...</span>
                    </Button>
                  )}
                  
                  {/* Show error button with full details */}
                  {hasError && (
                    <div className="w-full bg-red-50 p-3 rounded-lg border border-red-200">
                      <h4 className="text-sm font-medium text-red-800">Error Details</h4>
                      <p className="text-xs text-red-700 mt-1">{error}</p>
                    </div>
                  )}
                  
                  {/* Complete - close button only shown when complete */}
                  {(isComplete || hasError) && (
                    <Button
                      variant="default"
                      size="sm"
                      className="gap-1 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white rounded-full px-5"
                      onClick={() => forceCloseModal()}
                    >
                      <CheckCircle className="h-4 w-4" />
                      <span>Complete - Close</span>
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>,
        document.body
      )}
      
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
                  This will stop processing the current file. Processed records cannot be recovered.
                </p>
              </div>
            </div>
            
            <div className="flex justify-end gap-2 mt-4">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowCancelDialog(false)}
              >
                No, Continue
              </Button>
              <Button
                variant="destructive"
                size="sm"
                onClick={handleCancel}
              >
                Yes, Cancel Upload
              </Button>
            </div>
          </motion.div>
        </div>,
        document.body
      )}
      
      {/* Animations */}
      <style jsx global>{`
        @keyframes gradientShift {
          0% {
            background-position: 0% 50%;
          }
          100% {
            background-position: 100% 50%;
          }
        }
      `}</style>
    </AnimatePresence>
  );
}