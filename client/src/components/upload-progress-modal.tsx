import { motion } from "framer-motion";
import { 
  CheckCircle, 
  Clock, 
  Database, 
  FileText, 
  Loader2, 
  XCircle,
  AlertTriangle
} from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { createPortal } from "react-dom";
import { useEffect, useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { cancelUpload } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { 
  getUploadSessionId, 
  getUploadProgress, 
  isUploading as checkIsUploading,
  clearUploadState
} from "@/lib/upload-persistence";

export function UploadProgressModal() {
  const { isUploading, uploadProgress, setIsUploading, setUploadProgress } = useDisasterContext();
  const [highestProcessed, setHighestProcessed] = useState(0);
  const [isCancelling, setIsCancelling] = useState(false);
  const [showCancelDialog, setShowCancelDialog] = useState(false);
  const [isStalled, setIsStalled] = useState(false);
  const { toast } = useToast();
  
  // Stall detection references
  const stalledRetryCount = useRef(0);
  const lastProcessedTime = useRef(Date.now());
  
  // =======================================
  // Special effect to restore upload state on page refresh/reload
  // =======================================
  useEffect(() => {
    // Only try to restore if we're not already showing as uploading in the UI
    if (!isUploading) {
      try {
        // Check if an upload was in progress according to localStorage
        const wasUploading = checkIsUploading();
        const storedProgress = getUploadProgress();
        const sessionId = getUploadSessionId();
        
        if (wasUploading && storedProgress && sessionId) {
          console.log('Restoring upload progress after page refresh', storedProgress);
          
          // Immediately register the current window with the server to receive updates
          const connectToSSE = async () => {
            try {
              // Create EventSource for server-sent events to get progress updates
              const eventSource = new EventSource(`/api/upload-progress/${sessionId}`);
              
              eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('Received SSE progress update for restored session', data);
                setUploadProgress(data);
              };
              
              eventSource.onerror = () => {
                console.error('SSE connection error for restored session');
                eventSource.close();
              };
              
              // Set cleanup function
              return () => {
                console.log('Closing SSE connection for restored session');
                eventSource.close();
              };
            } catch (err) {
              console.error('Failed to connect to SSE for restored session', err);
            }
          };
          
          // Force the UI to show the upload modal with stored progress
          setIsUploading(true);
          setUploadProgress(storedProgress);
          
          // Establish SSE connection for live updates
          connectToSSE();
        }
      } catch (error) {
        console.error('Error recovering upload progress:', error);
      }
    }
  }, []);
  
  // =======================================
  // Effect to track the highest processed value (prevents jumpy progress)
  // =======================================
  useEffect(() => {
    if (uploadProgress.processed > 0 && uploadProgress.processed > highestProcessed) {
      setHighestProcessed(uploadProgress.processed);
      lastProcessedTime.current = Date.now();
      setIsStalled(false);
    }
  }, [uploadProgress.processed, highestProcessed]);

  // =======================================
  // Reset highest processed value when modal is closed 
  // =======================================
  useEffect(() => {
    if (!isUploading) {
      const timer = setTimeout(() => {
        setHighestProcessed(0);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isUploading]);

  // =======================================
  // Stalled detection with auto-recovery
  // =======================================
  useEffect(() => {
    // If processing but no progress change for 30 seconds, consider it stalled
    let stalledTimer: NodeJS.Timeout;
    const stalledCheckInterval = 10000; // Check every 10 seconds
    
    if (isUploading && !isStalled && uploadProgress.processed > 0 && uploadProgress.processed < uploadProgress.total) {
      stalledTimer = setInterval(() => {
        const now = Date.now();
        const timeSinceLastProgress = now - lastProcessedTime.current;
        
        // If no progress for 30 seconds
        if (timeSinceLastProgress > 30000) {
          setIsStalled(true);
          stalledRetryCount.current++;
          
          // After 3 retry attempts, suggest cancellation
          if (stalledRetryCount.current >= 3) {
            toast({
              title: "Upload appears to be stuck",
              description: "The upload has been stalled for a while. Consider canceling and trying again.",
              variant: "destructive"
            });
          }
        }
      }, stalledCheckInterval);
    }
    
    return () => {
      if (stalledTimer) clearInterval(stalledTimer);
    };
  }, [isUploading, isStalled, uploadProgress, toast]);
  
  // Don't render the modal if not uploading
  if (!isUploading) return null;
  
  const { 
    stage = 'Processing...', 
    processed: rawProcessed = 0, 
    total = 100,
    processingStats = {
      successCount: 0,
      errorCount: 0,
      lastBatchDuration: 0,
      averageSpeed: 0
    },
    currentSpeed = 0,
    timeRemaining = 0,
  } = uploadProgress;
  
  // Special handling for initial display to show something immediately
  const displayProcessed = Math.max(rawProcessed, stage.toLowerCase().includes('processing') && rawProcessed === 0 ? 1 : 0);
  
  // Only use highest recorded value for transitions to prevent progress going backward
  const processed = Math.max(displayProcessed, highestProcessed);

  // Calculate completion percentage safely with protection against NaN and extreme values  
  const percentComplete = total > 0 
    ? Math.min(100, Math.max(0, Math.round((processed / total) * 100)))
    : 0;
    
  // Different stage indicators
  const isBatchPause = stage.toLowerCase().includes('pause between batches');
  const isLoading = stage.toLowerCase().includes('loading') || stage.toLowerCase().includes('preparing');
  const isProcessing = (stage.toLowerCase().includes('processing') || stage.toLowerCase().includes('record')) && !isBatchPause;
  const isPaused = isBatchPause;
  
  // Completion detection
  const isComplete = (
    stage.toLowerCase().includes('complete') || 
    stage.toLowerCase().includes('analysis complete') ||
    (processed >= total && total > 0)
  );
  
  // Error detection
  const hasError = stage.toLowerCase().includes('error');
  
  // Handle cancel button click
  const handleCancel = async () => {
    if (isCancelling) return;
    
    setShowCancelDialog(false);
    setIsCancelling(true);
    
    try {
      const result = await cancelUpload();
      
      if (result.success) {
        // Clear all upload state from localStorage
        clearUploadState();
        
        toast({
          title: "Upload cancelled",
          description: "The upload has been cancelled successfully."
        });
        
        // Let the server-side events close the modal
      } else {
        setIsCancelling(false);
        toast({
          title: "Cancellation failed",
          description: result.message || "Failed to cancel upload. Please try again.",
          variant: "destructive"
        });
      }
    } catch (error) {
      setIsCancelling(false);
      toast({
        title: "Cancellation error",
        description: error instanceof Error ? error.message : "An unknown error occurred",
        variant: "destructive"
      });
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
            {isComplete ? 'Analysis Complete!' : hasError ? 'Upload Error' : isStalled ? 'Upload Stalled' : `Processing Records`}
          </h3>
          
          {/* Counter with larger size */}
          <div className="flex items-center justify-center my-3">
            <motion.div 
              key={processed}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ 
                opacity: 1, 
                scale: 1,
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
                ${isComplete ? 'text-green-600' : hasError ? 'text-red-600' : isStalled ? 'text-amber-600' : 'text-blue-600'}
              `}>
                {percentComplete}%
              </span>
            </div>
            <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
              <motion.div 
                className={`h-full ${
                  hasError 
                    ? 'bg-gradient-to-r from-red-400 to-red-600'
                    : isStalled
                      ? 'bg-gradient-to-r from-amber-400 to-amber-500'
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
                    : isStalled 
                      ? 'stalled-pulse 2s ease-in-out infinite'
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
                  <span className={`text-xs ml-1 ${isStalled ? 'text-amber-500' : isPaused ? 'text-amber-500' : 'text-green-500'}`}>
                    {isStalled ? '(stalled)' : isPaused ? '(paused)' : '(processing)'}
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
                    ? `Processing Records${isStalled ? ' (Stalled)' : ''}` 
                    : "Waiting to Process"}
              </span>
              {isProcessing && !isComplete && !isStalled && <Loader2 className="h-4 w-4 ml-auto animate-spin text-blue-500" />}
              {isProcessing && !isComplete && isStalled && <AlertTriangle className="h-4 w-4 ml-auto text-amber-500" />}
              {isComplete && <CheckCircle className="h-4 w-4 ml-auto text-green-500" />}
              {!isProcessing && !isComplete && <Clock className="h-4 w-4 ml-auto text-gray-400" />}
            </div>
          </div>
          
          {/* Stalled upload warning */}
          {isStalled && !isComplete && !hasError && (
            <div className="mb-4 p-3 bg-amber-50 border border-amber-100 rounded-lg text-amber-700">
              <div className="flex items-start gap-2">
                <div className="flex-shrink-0 mt-0.5">
                  <AlertTriangle className="h-4 w-4" />
                </div>
                <div>
                  <p className="text-xs font-medium">Upload appears to be stalled</p>
                  <p className="text-xs mt-1">You can wait or cancel and try again. Large files may take longer to process.</p>
                </div>
              </div>
            </div>
          )}
          
          {/* Processing statistics */}
          {!hasError && !isComplete && (
            <div className="grid grid-cols-2 gap-2 mb-4">
              {/* Processing rate */}
              <div className="bg-white/80 rounded-lg p-2 shadow-sm border border-blue-100">
                <div className="flex items-center gap-1 text-blue-600 mb-1">
                  <Clock className="h-3 w-3" />
                  <span className="text-[10px] font-medium uppercase">Processing Rate</span>
                </div>
                <div className="text-sm font-medium text-gray-700">
                  {processingStats.averageSpeed > 0 
                    ? `${processingStats.averageSpeed.toFixed(1)}/sec` 
                    : 'Calculating...'}
                </div>
              </div>
              
              {/* Batch info */}
              <div className="bg-white/80 rounded-lg p-2 shadow-sm border border-blue-100">
                <div className="flex items-center gap-1 text-blue-600 mb-1">
                  <Database className="h-3 w-3" />
                  <span className="text-[10px] font-medium uppercase">Success Rate</span>
                </div>
                <div className="text-sm font-medium text-gray-700">
                  {processingStats.successCount > 0 || processingStats.errorCount > 0
                    ? `${processingStats.successCount}/${processingStats.successCount + processingStats.errorCount}`
                    : 'Processing...'}
                </div>
              </div>
            </div>
          )}
          
          {/* Stage information */}
          <div className="mb-4 bg-white/90 rounded-lg p-3 shadow-sm border border-gray-100">
            <div className="text-sm font-medium text-gray-700">Status:</div>
            <div className={`text-sm ${hasError ? 'text-red-600 font-medium' : 'text-blue-700'}`}>
              {stage}
            </div>
          </div>
          
          {/* Cancel button & confirmation dialog */}
          {!isComplete && !hasError && (
            <>
              {/* Confirmation dialog */}
              {showCancelDialog && (
                <div className="mb-4">
                  <div className="bg-red-50 border border-red-100 p-3 rounded-lg text-red-700 mb-3">
                    <p className="text-sm font-medium">Are you sure you want to cancel?</p>
                    <p className="text-xs mt-1">This will stop all processing. Any data already processed will remain.</p>
                  </div>
                  <div className="flex justify-between gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-1/2"
                      onClick={() => setShowCancelDialog(false)}
                    >
                      No, Continue
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      className="w-1/2 bg-red-600"
                      onClick={handleCancel}
                      disabled={isCancelling}
                    >
                      {isCancelling ? (
                        <>
                          <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                          Cancelling...
                        </>
                      ) : (
                        'Yes, Cancel'
                      )}
                    </Button>
                  </div>
                </div>
              )}
              
              {/* Main cancel button */}
              {!showCancelDialog && (
                <div className="text-center">
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
            </>
          )}
          
          {/* Close button for completed or errored uploads */}
          {(isComplete || hasError) && (
            <div className="text-center">
              <Button
                variant={isComplete ? "default" : "destructive"}
                className={`gap-1 ${
                  isComplete 
                    ? "bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700" 
                    : "bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700"
                } text-white font-medium rounded-full px-5 py-2`}
                onClick={() => {
                  // Clear all upload state from localStorage on close
                  clearUploadState();
                  
                  // Close the modal
                  setIsUploading(false);
                  
                  // Reset progress
                  setUploadProgress({
                    processed: 0,
                    total: 0,
                    stage: '',
                  });
                }}
              >
                {isComplete 
                  ? <CheckCircle className="h-4 w-4 mr-1" /> 
                  : <XCircle className="h-4 w-4 mr-1" />}
                {isComplete ? "Close" : "Dismiss"}
              </Button>
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>,
    document.body
  );
}