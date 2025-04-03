import { AnimatePresence, motion } from "framer-motion";
import { Loader2, FileText, Database, ChevronRight, Activity, Clock, AlertTriangle, XCircle } from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { createPortal } from "react-dom";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { cancelUpload } from "@/lib/api";

// Animated number component for smooth transitions
const AnimatedNumber = ({ value }: { value: number }) => (
  <motion.span
    key={value}
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -10 }}
    transition={{ duration: 0.3, ease: "easeOut" }}
    className="tabular-nums"
  >
    {value.toLocaleString()}
  </motion.span>
);

// Format time remaining
const formatTimeRemaining = (seconds: number): string => {
  if (!seconds || seconds <= 0) return '';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`;
};

// Format speed
const formatSpeed = (recordsPerSecond: number): string => {
  if (!recordsPerSecond || recordsPerSecond <= 0) return '';
  if (recordsPerSecond >= 1000) {
    return `${(recordsPerSecond / 1000).toFixed(1)}k records/s`;
  }
  return `${Math.round(recordsPerSecond)} records/s`;
};

export function UploadProgressModal() {
  const { isUploading, uploadProgress, setIsUploading } = useDisasterContext();
  const [highestProcessed, setHighestProcessed] = useState(0);
  const [isCancelling, setIsCancelling] = useState(false);

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
    
    setIsCancelling(true);
    try {
      const result = await cancelUpload();
      console.log('Cancel upload result:', result);
      
      if (result.success) {
        // We'll let the server-side events close the modal
        // The progress will be updated to show "Upload cancelled by user"
      } else {
        // If the server couldn't cancel (maybe already finished), close the modal
        setIsUploading(false);
      }
    } catch (error) {
      console.error('Error cancelling upload:', error);
      // Still close the modal to let the user try again
      setIsUploading(false);
    } finally {
      setIsCancelling(false);
    }
  };

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
    batchProgress = 0
  } = uploadProgress;

  // Use the higher value between current processed and highest recorded
  const processed = Math.max(rawProcessed, highestProcessed);

  // Calculate completion percentage
  const percentComplete = Math.min(Math.round((processed / total) * 100), 100);

  // Stage indication
  const isLoading = stage.toLowerCase().includes('loading');
  const isProcessing = stage.toLowerCase().includes('processing') || stage.toLowerCase().includes('record');
  const isCooldown = stage.toLowerCase().includes('cooldown') || stage.toLowerCase().includes('pausing') || stage.toLowerCase().includes('pause between batches');
  const hasError = stage.toLowerCase().includes('error');
  const isComplete = stage.toLowerCase().includes('analysis complete');
  
  // Extract cooldown time remaining if applicable
  const cooldownTimeRemaining = (() => {
    if (isCooldown) {
      // First check for explicit cooldown time in message
      const match = stage.match(/Cooldown: (\d+) seconds? remaining/i) || 
                    stage.match(/(\d+) seconds? remaining/i) ||
                    stage.match(/(\d+)-second pause/i);
      
      if (match && match[1]) {
        return parseInt(match[1]);
      }
      
      // If it's a cooldown but no time specified, default to 60 seconds
      if (stage.toLowerCase().includes('pause between batches')) {
        return 60;
      }
    }
    return null;
  })();

  return createPortal(
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="fixed inset-0 flex items-center justify-center z-[9999]"
    >
      {/* Enhanced backdrop with animated gradient */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-indigo-600/20 to-purple-600/20 backdrop-blur-lg"
      />

      {/* Content */}
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="relative bg-gradient-to-br from-slate-900/95 via-slate-800/95 to-slate-900/95 backdrop-blur-lg rounded-xl border border-slate-700/50 p-8 max-w-md w-full mx-4 shadow-2xl"
      >
        {/* Animated background pattern */}
        <div className="absolute inset-0 overflow-hidden rounded-xl">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-purple-500/5 to-indigo-500/5 animate-gradient"></div>
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(59,130,246,0.08),transparent_50%)]"></div>
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_bottom_left,rgba(147,51,234,0.08),transparent_50%)]"></div>
          <div className="absolute inset-0 bg-grid-slate-700/20"></div>
        </div>

        {/* Content wrapper */}
        <div className="relative space-y-2">
          {/* Main Progress Display */}
          <div className="text-center mb-2">
            <h3 className="text-xl font-semibold text-slate-200 mb-1">
              {hasError ? 'Upload Error' :
                isComplete 
                  ? 'Analysis Complete!' 
                  : total > 0 
                    ? `Processing ${total.toLocaleString()} records...` 
                    : 'Preparing upload...'}
            </h3>
            {!hasError && (
              <div className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent flex items-center justify-center gap-2">
                <AnimatedNumber value={processed} />
                <span className="text-slate-500">/</span>
                <AnimatedNumber value={total} />
              </div>
            )}
            {hasError && (
              <div className="mt-2 flex items-center justify-center gap-2 text-red-400">
                <AlertTriangle className="h-5 w-5" />
                <span className="text-sm">An error occurred during upload. Please try again.</span>
              </div>
            )}
          </div>

          {/* Cooldown Timer Display */}
          {isCooldown && cooldownTimeRemaining !== null && (
            <div className="mb-4 bg-gradient-to-br from-amber-800/30 to-amber-700/30 backdrop-blur-sm rounded-xl border border-amber-600/30 overflow-hidden shadow-lg">
              <div className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-full bg-amber-500/20">
                      <Clock className="h-5 w-5 text-amber-400 animate-pulse" />
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold text-amber-300">60-Second Cooldown Period</h4>
                      <p className="text-xs text-amber-300/70">Processing will resume automatically</p>
                    </div>
                  </div>
                  <div className="text-2xl font-bold font-mono text-amber-300 tabular-nums">
                    <motion.div
                      key={cooldownTimeRemaining}
                      initial={{ scale: 1.2, opacity: 0.7 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ duration: 0.2 }}
                    >
                      {cooldownTimeRemaining}s
                    </motion.div>
                  </div>
                </div>
                
                {/* Countdown progress bar */}
                <div className="h-3 bg-slate-800/50 rounded-full overflow-hidden">
                  <motion.div 
                    className="h-full bg-gradient-to-r from-amber-400 to-amber-500"
                    initial={{ width: '100%' }}
                    animate={{ 
                      width: '0%',
                      transition: { duration: cooldownTimeRemaining, ease: 'linear' }
                    }}
                  />
                </div>
                
                <div className="mt-3 text-xs text-amber-300/70 italic text-center">
                  Required 60-second pause between batches<br />
                  <span className="font-semibold">Processing in batches of 30 records</span>
                </div>
              </div>
            </div>
          )}

          {/* Enhanced Stats Grid */}
          {!hasError && !isCooldown && (
            <div className="grid grid-cols-2 gap-4 mb-4">
              {/* Processing Speed */}
              {(currentSpeed > 0 || processingStats.averageSpeed > 0) && (
                <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-xl p-4 border border-slate-600/50 shadow-lg">
                  <div className="flex items-center gap-2 text-blue-400 mb-2">
                    <Activity className="h-4 w-4" />
                    <span className="text-xs font-medium">Processing Speed</span>
                  </div>
                  <div className="text-sm font-semibold text-slate-300">
                    {currentSpeed > 0 ? formatSpeed(currentSpeed) : formatSpeed(processingStats.averageSpeed)}
                  </div>
                </div>
              )}

              {/* Time Remaining */}
              {timeRemaining > 0 && !isComplete && (
                <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-xl p-4 border border-slate-600/50 shadow-lg">
                  <div className="flex items-center gap-2 text-blue-400 mb-2">
                    <Clock className="h-4 w-4" />
                    <span className="text-xs font-medium">Time Remaining</span>
                  </div>
                  <div className="text-sm font-semibold text-slate-300">
                    {formatTimeRemaining(timeRemaining)}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Progress Stages */}
          <div className="space-y-2">
            {/* Loading Stage */}
            <div className={`flex items-center gap-3 p-3 rounded-lg transition-colors ${
              isLoading ? 'bg-blue-900/30 text-blue-300 shadow-lg border border-blue-500/20' : 'bg-slate-800/50 text-slate-400'
            }`}>
              <FileText className="h-4 w-4" />
              <span className="text-sm font-medium">Loading File</span>
              {isLoading && <Loader2 className="h-4 w-4 ml-auto animate-spin" />}
              {!isLoading && <ChevronRight className="h-4 w-4 ml-auto" />}
            </div>

            {/* Processing Stage */}
            <div className={`flex items-center gap-3 p-3 rounded-lg transition-colors ${
              isProcessing ? 'bg-blue-900/30 text-blue-300 shadow-lg border border-blue-500/20' : 'bg-slate-800/50 text-slate-400'
            }`}>
              <Database className="h-4 w-4" />
              <span className="text-sm font-medium">
                {isProcessing ? `Processing record ${processed.toLocaleString()} of ${total.toLocaleString()}` : "Processing Records"}
              </span>
              {isProcessing && <Loader2 className="h-4 w-4 ml-auto animate-spin" />}
              {!isProcessing && <ChevronRight className="h-4 w-4 ml-auto" />}
            </div>

            {/* Status Message */}
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <div className="text-sm font-mono text-slate-300 whitespace-pre-wrap break-words">
                {stage}
              </div>
            </div>
          </div>

          {/* Enhanced Progress Bar */}
          <div>
            <div className="flex justify-between text-sm font-medium mb-2">
              <span className="text-slate-300">Overall Progress</span>
              <span className={`transition-colors ${isComplete ? 'text-green-400' : hasError ? 'text-red-400' : 'text-blue-400'}`}>
                {percentComplete}%
              </span>
            </div>
            <div className="h-3 bg-slate-800/50 rounded-full overflow-hidden shadow-inner border border-slate-700/50">
              <motion.div 
                className={`h-full transition-colors duration-300 ${
                  hasError 
                    ? 'bg-gradient-to-r from-red-500 to-red-600'
                    : isComplete
                      ? 'bg-gradient-to-r from-green-400 to-emerald-500' 
                      : 'bg-gradient-to-r from-blue-400 via-blue-500 to-indigo-500'
                }`}
                initial={{ width: 0 }}
                animate={{ 
                  width: `${percentComplete}%`,
                  transition: { duration: 0.3 }
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
          
          {/* Cancel Button - only show when processing and not during cooldown or when complete */}
          {!isComplete && !isCooldown && (
            <div className="mt-6 text-center">
              <Button
                variant="destructive"
                size="sm"
                className="gap-1 bg-red-900/80 hover:bg-red-800 text-white"
                onClick={handleCancel}
                disabled={isCancelling || isComplete}
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

          {/* Enhanced animations */}
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

              @keyframes gradient {
                0% {
                  background-position: 0% 50%;
                }
                50% {
                  background-position: 100% 50%;
                }
                100% {
                  background-position: 0% 50%;
                }
              }

              .animate-gradient {
                background-size: 200% 200%;
                animation: gradient 8s ease infinite;
              }

              .bg-grid-slate-700/20 {
                background-size: 30px 30px;
                background-image: linear-gradient(to right, rgba(51, 65, 85, 0.1) 1px, transparent 1px),
                                 linear-gradient(to bottom, rgba(51, 65, 85, 0.1) 1px, transparent 1px);
              }
            `}
          </style>
        </div>
      </motion.div>
    </motion.div>,
    document.body
  );
}