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
  const { isUploading, uploadProgress, setIsUploading } = useDisasterContext();
  const [isCancelling, setIsCancelling] = useState(false);
  const [showCancelDialog, setShowCancelDialog] = useState(false);
  
  // Handle cancel button click
  const handleCancel = async () => {
    if (isCancelling) return;
    
    setShowCancelDialog(false);
    setIsCancelling(true);
    try {
      const result = await cancelUpload();
      
      if (result.success) {
        // We'll let the server-side events close the modal
      } else {
        setIsCancelling(false);
      }
    } catch (error) {
      setIsCancelling(false);
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
  
  // SIMPLIFIED STAGE DETECTION LOGIC
  // Convert stage to lowercase once for all checks
  const stageLower = stage.toLowerCase();
  
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
  
  // Calculate time remaining in human-readable format
  const formatTimeRemaining = (seconds: number): string => {
    if (!seconds || seconds <= 0) return 'calculating...';
    if (seconds < 60) return `${Math.ceil(seconds)} sec`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.ceil(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
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
            
            {/* Progress bar */}
            <div className="w-full mt-4 mb-1">
              <div className="h-2 bg-black/10 rounded-full overflow-hidden relative">
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
              </div>
              <div className="flex justify-between text-xs mt-1">
                <span className="text-white/70">{Math.floor(percentComplete)}%</span>
                <span className="text-white/70">
                  {currentSpeed > 0 ? `${currentSpeed.toFixed(1)} records/sec` : ''}
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

            {/* Processing stats */}
            {isProcessing && !hasError && (
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
            )}
            
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
                    onClick={() => setIsUploading(false)}
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
                  onClick={() => setIsUploading(false)}
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