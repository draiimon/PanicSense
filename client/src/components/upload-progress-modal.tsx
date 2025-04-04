import { motion } from "framer-motion";
import { 
  CheckCircle, 
  Clock, 
  Database, 
  FileText, 
  Loader2, 
  XCircle
} from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { createPortal } from "react-dom";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { cancelUpload } from "@/lib/api";
// removed import to fix errors

// Simple helper functions for upload progress

export function UploadProgressModal() {
  const { isUploading, uploadProgress, setIsUploading } = useDisasterContext();
  const [isCancelling, setIsCancelling] = useState(false);
  const [showCancelDialog, setShowCancelDialog] = useState(false);
  
  // Fixed approach for animation with proper conditions
  const breathingOffset = { scale: 1 };

  // No need to track highest processed value anymore - we're showing the raw value
  // and using fixed height to prevent layout shifts

  // Since we now let the raw count show directly, we don't need to track highest value anymore
  
  // Handle cancel button click
  const handleCancel = async () => {
    if (isCancelling) return;
    
    setShowCancelDialog(false);
    setIsCancelling(true);
    try {
      const result = await cancelUpload();
      // Removed console.log as requested
      
      if (result.success) {
        // We'll let the server-side events close the modal
      } else {
        setIsCancelling(false);
      }
    } catch (error) {
      // Removed console.error as requested
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
  
  // Use the actual processed value directly from the server
  // Only for display consistency, don't artificially inflate initial value
  const processed = rawProcessed;
  
  // Min-height styling to prevent UI shrinking during transitions
  const modalContentStyle = {
    minHeight: '420px' // Set min-height to prevent modal shrinking
  };

  // Calculate completion percentage safely
  const percentComplete = total > 0 
    ? Math.min(100, Math.round((processed / total) * 100)) 
    : 0;

  // More sophisticated stage indication with better handling
  const isBatchPause = stage.toLowerCase().includes('pause between batches');
  const isLoading = stage.toLowerCase().includes('loading') || stage.toLowerCase().includes('preparing');
  const isProcessing = (stage.toLowerCase().includes('processing') || stage.toLowerCase().includes('record')) && !isBatchPause;
  const isPaused = isBatchPause;
  
  // Better completion detection with multiple triggers
  const isComplete = (
    stage.toLowerCase().includes('complete') || 
    stage.toLowerCase().includes('analysis complete') ||
    (processed >= total && total > 0)
  );
  
  // Improved error detection
  const hasError = stage.toLowerCase().includes('error');

  return createPortal(
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="fixed inset-0 flex items-center justify-center z-[9999]"
    >
      {/* Gradient backdrop */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/30 via-indigo-400/20 to-purple-500/30 backdrop-blur-md"></div>

      {/* Content */}
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        transition={{ duration: 0.2 }}
        style={{
          background: "rgba(255, 255, 255, 0.85)",
          boxShadow: "0 8px 32px rgba(31, 38, 135, 0.15)",
          backdropFilter: "blur(8px)",
          border: "1px solid rgba(255, 255, 255, 0.2)",
          minHeight: "480px", // Fixed height to prevent shrinking
        }}
        className="relative rounded-xl overflow-hidden w-full max-w-sm mx-4"
      >
        {/* Gradient Header */}
        <div className="bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-500 p-4 text-white">
          <h3 className="text-xl font-bold text-center">
            {isComplete ? 'Analysis Complete!' : hasError ? 'Upload Error' : `Processing Records`}
          </h3>
          
          {/* Counter with larger size */}
          <div className="flex items-center justify-center my-3">
            <motion.div 
              key={processed}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ 
                opacity: 1, 
                scale: isProcessing ? breathingOffset.scale : 1,
                transition: { duration: 0.5 }
              }}
              className="flex flex-col items-center"
            >
              <div className="flex items-center">
                <span className="text-5xl font-bold text-white">{processed}</span>
                <span className="text-3xl mx-2 text-white/70">/</span>
                <span className="text-4xl font-bold text-white/90">{total}</span>
              </div>
              <span className="text-sm text-white/80 mt-1">Records Processed</span>
            </motion.div>
          </div>
        </div>
        
        <div className="p-5">
          {/* Enhanced Progress Bar */}
          <div className="mb-4">
            <div className="flex justify-between text-sm font-medium mb-1">
              <span className="text-gray-600">Overall Progress</span>
              <span className={`font-medium
                ${isComplete ? 'text-green-600' : hasError ? 'text-red-600' : 'text-blue-600'}
              `}>
                {percentComplete}%
              </span>
            </div>
            <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
              <motion.div 
                className={`h-full ${
                  hasError 
                    ? 'bg-gradient-to-r from-red-400 to-red-600'
                    : isComplete
                      ? 'bg-gradient-to-r from-green-400 to-emerald-500' 
                      : 'bg-gradient-to-r from-blue-400 via-indigo-500 to-purple-500'
                }`}
                initial={{ width: 0 }}
                animate={{ 
                  width: `${percentComplete}%`,
                  transition: { duration: 0.5 }
                }}
                style={{
                  backgroundSize: '200% 100%',
                  animation: isComplete
                    ? 'completion-pulse 2s ease-in-out infinite' 
                    : 'progress-flow 2s linear infinite'
                }}
              />
            </div>
          </div>
          
          {/* Enhanced Real-time Processing Stats with improved UI */}
          {!hasError && !isComplete && (
            <div className="grid grid-cols-1 gap-3 mb-4">               
              {/* Records Remaining with dynamic display */}
              <div className="bg-white/80 rounded-lg p-3 shadow-sm border border-blue-100 hover:shadow-md transition-all">
                <div className="flex items-center gap-2 text-blue-600 mb-1">
                  <Database className="h-4 w-4" />
                  <span className="text-xs font-medium">Records Remaining</span>
                </div>
                <div className="text-sm font-bold text-gray-700">
                  {Math.max(0, total - processed)}
                  <span className={`text-xs ml-1 ${isPaused ? 'text-amber-500' : 'text-green-500'}`}>
                    {isPaused ? '(paused)' : '(processing)'}
                  </span>
                </div>
              </div>
            </div>
          )}
          
          {/* Error display section */}
          {hasError && (
            <div className="my-4 px-3">
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <div className="flex items-center gap-2 text-red-600 mb-2">
                  <XCircle className="h-5 w-5" />
                  <h4 className="font-medium">Upload Error</h4>
                </div>
                <p className="text-sm text-red-700">{error || 'There was an error processing your upload. Please try again.'}</p>
              </div>
              <div className="mt-4 text-center">
                <Button
                  onClick={() => setIsUploading(false)}
                  className="bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700 text-white"
                >
                  Close
                </Button>
              </div>
            </div>
          )}
          
          {/* Processing Steps - Only show when no error */}
          {!hasError && (
            <div className="space-y-2 mb-4">
              {/* Loading Stage */}
              <div className={`flex items-center gap-3 p-2 rounded-lg transition-all ${
                isLoading 
                  ? 'bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-700 border border-blue-100' 
                  : 'bg-gray-50 text-gray-500 border border-gray-100'
              }`}>
                <div className={`flex items-center justify-center w-6 h-6 rounded-full 
                  ${isLoading ? 'bg-blue-100' : 'bg-gray-200'}`
                }>
                  <FileText className="h-3 w-3" />
                </div>
                <span className="text-sm font-medium">File Preparation</span>
                {isLoading && <Loader2 className="h-4 w-4 ml-auto animate-spin text-blue-500" />}
                {!isLoading && <CheckCircle className="h-4 w-4 ml-auto text-green-500" />}
              </div>
              
              {/* Processing Stage */}
              <div className={`flex items-center gap-3 p-2 rounded-lg transition-all ${
                isProcessing && !isComplete
                  ? 'bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-700 border border-blue-100' 
                  : isComplete
                    ? 'bg-gradient-to-r from-green-50 to-emerald-50 text-green-700 border border-green-100'
                    : 'bg-gray-50 text-gray-500 border border-gray-100'
              }`}>
                <div className={`flex items-center justify-center w-6 h-6 rounded-full 
                  ${isProcessing && !isComplete 
                    ? 'bg-blue-100' 
                    : isComplete 
                      ? 'bg-green-100' 
                      : 'bg-gray-200'}`
                }>
                  <Database className="h-3 w-3" />
                </div>
                <span className="text-sm font-medium">
                  {isComplete 
                    ? "Processing Complete" 
                    : isProcessing 
                      ? `Processing Records` 
                      : "Waiting to Process"}
                </span>
                {isProcessing && !isComplete && <Loader2 className="h-4 w-4 ml-auto animate-spin text-blue-500" />}
                {isComplete && <CheckCircle className="h-4 w-4 ml-auto text-green-500" />}
                {!isProcessing && !isComplete && <Clock className="h-4 w-4 ml-auto text-gray-400" />}
              </div>
              
              {/* Batch Processing Info */}
              {isProcessing && batchNumber && totalBatches && (
                <div className="flex items-center gap-3 p-2 rounded-lg bg-gradient-to-r from-purple-50 to-blue-50 text-purple-700 border border-purple-100">
                  <div className="flex items-center justify-center w-6 h-6 rounded-full bg-purple-100">
                    <Database className="h-3 w-3" />
                  </div>
                  <span className="text-sm font-medium">
                    Saving Batch {batchNumber} of {totalBatches}
                  </span>
                  <Loader2 className="h-4 w-4 ml-auto animate-spin text-purple-500" />
                </div>
              )}
            </div>
          )}
          
          {/* Cancel button */}
          {!isComplete && !hasError && (
            <div className="mt-3 text-center">
              <Button
                variant="destructive"
                size="sm"
                className="gap-1 bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700 text-white font-medium rounded-full px-5 py-2"
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
        </div>
        
        {/* Animated patterns */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-blue-500/20 to-transparent rounded-full -mr-10 -mt-10"></div>
          <div className="absolute bottom-0 left-0 w-16 h-16 bg-gradient-to-tr from-purple-500/20 to-transparent rounded-full -ml-8 -mb-8"></div>
        </div>
      </motion.div>
      
      {/* Prettier Cancel Confirmation Dialog */}
      {showCancelDialog && createPortal(
        <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-[10000]" onClick={() => setShowCancelDialog(false)}>
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="bg-white rounded-xl p-4 max-w-xs mx-4 shadow-xl border border-gray-200" 
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-red-100 p-2 rounded-full">
                <XCircle className="h-5 w-5 text-red-500" />
              </div>
              <h3 className="text-lg font-bold text-gray-800">Cancel Upload?</h3>
            </div>
            <p className="text-gray-600 text-sm mb-4">
              All progress will be lost and you'll need to start the upload again.
            </p>
            <div className="flex justify-end gap-2">
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setShowCancelDialog(false)}
                className="bg-gray-50 text-gray-700 hover:bg-gray-100 hover:text-gray-900 border-gray-200 rounded-full px-4"
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
          @keyframes progress-flow {
            0% {
              background-position: 100% 0;
            }
            100% {
              background-position: -100% 0;
            }
          }

          @keyframes completion-pulse {
            0% {
              opacity: 0.8;
              transform: scale(1);
            }
            50% {
              opacity: 1;
              transform: scale(1.02);
            }
            100% {
              opacity: 0.8;
              transform: scale(1);
            }
          }
        `}
      </style>
    </motion.div>,
    document.body
  );
}