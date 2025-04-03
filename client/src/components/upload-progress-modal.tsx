import { motion } from "framer-motion";
import { Activity, BarChart3, CheckCircle, Clock, Database, FileText, Loader2, XCircle } from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { createPortal } from "react-dom";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { cancelUpload } from "@/lib/api";

// Helper to format processing speed in a human-readable format
const formatSpeed = (recordsPerSecond: number): string => {
  if (recordsPerSecond >= 1000) {
    return `${(recordsPerSecond / 1000).toFixed(1)}k records/s`;
  }
  return `${Math.round(recordsPerSecond)} records/s`;
};

// Helper to format time remaining
const formatTimeRemaining = (seconds: number): string => {
  if (seconds < 60) {
    return `${Math.ceil(seconds)} seconds`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.ceil(seconds % 60);
  return `${minutes}m ${remainingSeconds}s`;
};

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

  // Reset highest processed value when modal is closed
  useEffect(() => {
    if (!isUploading) {
      setHighestProcessed(0);
    }
  }, [isUploading]);
  
  // Handle cancel button click
  const handleCancel = async () => {
    if (isCancelling) return;
    
    setShowCancelDialog(false);
    setIsCancelling(true);
    try {
      const result = await cancelUpload();
      console.log('Cancel upload result:', result);
      
      if (result.success) {
        // We'll let the server-side events close the modal
      } else {
        setIsCancelling(false);
      }
    } catch (error) {
      console.error('Error cancelling upload:', error);
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

  // Simplified stage indication
  const isBatchPause = stage.toLowerCase().includes('pause between batches');
  const isLoading = stage.toLowerCase().includes('loading') || stage.toLowerCase().includes('preparing');
  const isProcessing = (stage.toLowerCase().includes('processing') || stage.toLowerCase().includes('record')) && !isBatchPause;
  const isPaused = isBatchPause;
  const isComplete = stage.toLowerCase().includes('complete');
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
          
          {/* Simple Processing Stats - show only relevant info */}
          {!hasError && !isComplete && (
            <div className="grid grid-cols-2 gap-3 mb-4">
              {/* Records Progress */}
              <div className="bg-white/90 rounded-lg p-3 shadow-sm border border-gray-100">
                <div className="flex items-center gap-2 text-blue-600 mb-1">
                  <Database className="h-4 w-4" />
                  <span className="text-xs font-medium">Records Processed</span>
                </div>
                <div className="text-sm font-bold text-gray-700">
                  {processed} of {total} ({percentComplete}%)
                </div>
              </div>
              
              {/* Records Remaining */}
              <div className="bg-white/90 rounded-lg p-3 shadow-sm border border-gray-100">
                <div className="flex items-center gap-2 text-purple-600 mb-1">
                  <Activity className="h-4 w-4" />
                  <span className="text-xs font-medium">Records Remaining</span>
                </div>
                <div className="text-sm font-bold text-gray-700">
                  {Math.max(0, total - processed)}
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
              {isProcessing && !isComplete && <Loader2 className="h-4 w-4 ml-auto animate-spin text-blue-500" />}
              {isComplete && <CheckCircle className="h-4 w-4 ml-auto text-green-500" />}
              {!isProcessing && !isComplete && <Clock className="h-4 w-4 ml-auto text-gray-400" />}
            </div>
          </div>
          
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