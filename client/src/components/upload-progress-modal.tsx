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
  const [highestProcessed, setHighestProcessed] = useState(0);
  const [isCancelling, setIsCancelling] = useState(false);
  const [showCancelDialog, setShowCancelDialog] = useState(false);
  
  // Fixed approach for animation with proper conditions
  const breathingOffset = { scale: 1 };

  // Effect to track the highest processed value
  useEffect(() => {
    if (uploadProgress.processed > 0 && uploadProgress.processed > highestProcessed) {
      setHighestProcessed(uploadProgress.processed);
    }
  }, [uploadProgress.processed, highestProcessed]);

  // Reset highest processed value when modal is closed plus smoother transitions
  useEffect(() => {
    if (!isUploading) {
      // Add a small delay to prevent any jump/flash effect during state transitions 
      const timer = setTimeout(() => {
        setHighestProcessed(0);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isUploading]);
  
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
    currentSpeed = 0,
    timeRemaining = 0,
  } = uploadProgress;
  
  // Simplified special handling for initial display
  const displayProcessed = Math.max(rawProcessed, stage.toLowerCase().includes('processing') && rawProcessed === 0 ? 1 : 0);
  
  // Only use highest recorded value for transitions, not for actual counts
  const processed = Math.max(displayProcessed, highestProcessed);

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

      {/* Cancel Confirmation Dialog */}
      {showCancelDialog && (
        <div className="absolute z-20 inset-0 flex items-center justify-center bg-black/20">
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-white rounded-xl p-5 shadow-xl max-w-md mx-4"
          >
            <h3 className="text-lg font-bold text-gray-800 mb-2">Cancel Upload?</h3>
            <p className="text-gray-600 mb-4">
              Are you sure you want to cancel this upload? The process will be stopped immediately.
            </p>
            <div className="flex justify-end gap-3">
              <Button 
                variant="outline"
                onClick={() => setShowCancelDialog(false)}
                disabled={isCancelling}
              >
                Continue Upload
              </Button>
              <Button 
                variant="destructive"
                onClick={handleCancel}
                disabled={isCancelling}
              >
                {isCancelling ? 'Cancelling...' : 'Cancel Upload'}
              </Button>
            </div>
          </motion.div>
        </div>
      )}

      {/* Main Modal */}
      <motion.div 
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 20, opacity: 0 }}
        transition={{ delay: 0.1, duration: 0.3 }}
        className="relative z-10 max-w-xl w-full mx-4 bg-white rounded-2xl shadow-2xl overflow-hidden"
      >
        {/* Header with status indicator */}
        <div className={`px-6 py-4 flex items-center justify-between border-b ${
          hasError 
            ? 'bg-red-50 border-red-100' 
            : isComplete 
              ? 'bg-green-50 border-green-100' 
              : 'bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-100'
        }`}>
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-full ${
              hasError 
                ? 'bg-red-100 text-red-600' 
                : isComplete 
                  ? 'bg-green-100 text-green-600' 
                  : 'bg-blue-100 text-blue-600'
            }`}>
              {hasError && <XCircle className="h-5 w-5" />}
              {isComplete && <CheckCircle className="h-5 w-5" />}
              {!hasError && !isComplete && (
                <motion.div 
                  className="p-0.5 flex items-center justify-center"
                  animate={breathingOffset}
                  transition={{ duration: 1.5, repeat: Infinity, repeatType: "reverse" }}
                >
                  <Loader2 className="h-5 w-5 animate-spin" />
                </motion.div>
              )}
            </div>
            <div>
              <h3 className={`font-semibold leading-none ${
                hasError 
                  ? 'text-red-700' 
                  : isComplete 
                    ? 'text-green-700' 
                    : 'text-blue-700'
              }`}>
                {hasError 
                  ? 'Upload Error' 
                  : isComplete 
                    ? 'Upload Complete' 
                    : 'Uploading Data'
                }
              </h3>
              <p className="text-sm text-gray-500 mt-1">
                {stage}
              </p>
            </div>
          </div>
          
          {/* Cancel/Close Button with Conditional Rendering */}
          {!isCancelling && ( 
            <Button
              variant="ghost"
              size="icon"
              onClick={() => isComplete ? setIsUploading(false) : setShowCancelDialog(true)}
              disabled={isCancelling}
              className={`rounded-full h-8 w-8 ${
                isComplete 
                  ? 'hover:bg-green-100 hover:text-green-600' 
                  : 'hover:bg-red-100 hover:text-red-600'
              }`}
            >
              <XCircle className={`h-5 w-5 ${isComplete ? 'text-green-500' : 'text-gray-400'}`} />
            </Button>
          )}
        </div>

        {/* Body Content */}
        <div className="p-6">
          {/* Progress Display with Motion */}
          <div className="mb-5">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium">
                {processed} of {total} records processed
              </span>
              <span className="text-sm font-medium">
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
          
          {/* Processing Steps */}
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
              {isProcessing && !isComplete && (
                <Loader2 className="h-4 w-4 ml-auto animate-spin text-blue-500" />
              )}
              {isComplete && <CheckCircle className="h-4 w-4 ml-auto text-green-500" />}
              {!isProcessing && !isComplete && <Clock className="h-4 w-4 ml-auto text-gray-400" />}
            </div>
          </div>
          
          {/* Completion or Error Message */}
          {isComplete && (
            <div className="bg-green-50 border border-green-100 rounded-lg p-4 text-center mb-4">
              <CheckCircle className="h-6 w-6 mx-auto text-green-500 mb-2" />
              <h4 className="font-medium text-green-800">Processing Complete</h4>
              <p className="text-sm text-gray-600 mt-1">
                Your data has been processed successfully.
              </p>
            </div>
          )}
          
          {hasError && (
            <div className="bg-red-50 border border-red-100 rounded-lg p-4 text-center mb-4">
              <XCircle className="h-6 w-6 mx-auto text-red-500 mb-2" />
              <h4 className="font-medium text-red-800">Processing Error</h4>
              <p className="text-sm text-gray-600 mt-1">
                {uploadProgress?.error || "An error occurred during processing."}
              </p>
            </div>
          )}
          
          {/* Action Buttons */}
          <div className="flex justify-end gap-2">
            {!hasError && !isComplete && !isCancelling && (
              <Button
                onClick={() => setShowCancelDialog(true)}
                className="bg-gray-50 text-gray-700 hover:bg-gray-100 hover:text-gray-900 border-gray-200"
              >
                Cancel Upload
              </Button>
            )}
            
            {isComplete && (
              <Button
                onClick={() => setIsUploading(false)}
                className="bg-green-50 text-green-700 hover:bg-green-100 border-green-200"
              >
                Close
              </Button>
            )}
            
            {hasError && (
              <Button
                onClick={() => setIsUploading(false)}
                className="bg-red-50 text-red-700 hover:bg-red-100 border-red-200"
              >
                Close
              </Button>
            )}
          </div>
        </div>
      </motion.div>
    </motion.div>,
    document.body
  );
}