import { AnimatePresence, motion } from "framer-motion";
import { Loader2, FileText, Database, ChevronRight, Activity, Clock, AlertTriangle } from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { createPortal } from "react-dom";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useEffect, useState, useRef } from "react";

// Animated number component for smooth transitions
const AnimatedNumber = ({ value }: { value: number }) => (
  <motion.span
    key={value}
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -20 }}
    transition={{ duration: 0.2 }}
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
  const { isUploading, uploadProgress } = useDisasterContext();
  const [activeTab, setActiveTab] = useState<'progress'>('progress');
  const [highestProcessed, setHighestProcessed] = useState(0);

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
  const isLoading = stage.toLowerCase().includes('initializing') || stage.toLowerCase().includes('loading');
  const isProcessing = stage.toLowerCase().includes('processing') || stage.toLowerCase().includes('record');
  const isCompleted = stage.toLowerCase().includes('complete') || stage.toLowerCase().includes('completed');

  return createPortal(
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="fixed inset-0 flex items-center justify-center z-[9999]"
    >
      {/* Enhanced backdrop with subtle animation */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        className="absolute inset-0 bg-gradient-to-br from-black/30 to-black/10 backdrop-blur-sm"
      />

      {/* Content */}
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="relative bg-white/95 backdrop-blur-lg rounded-xl border border-blue-100/50 p-8 max-w-md w-full mx-4 shadow-2xl"
      >
        {/* Animated background pattern */}
        <div className="absolute inset-0 overflow-hidden rounded-xl">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-50/50 via-purple-50/50 to-indigo-50/50 animate-gradient"></div>
          <div className="absolute inset-0 bg-grid-slate-200/50"></div>
        </div>

        {/* Content wrapper */}
        <div className="relative space-y-6">
          {/* Main Progress Display */}
          <div className="text-center">
            <h3 className="text-lg font-semibold text-slate-800 mb-2">
              {stage.toLowerCase().includes('analysis complete') 
                ? 'Analysis Complete!' 
                : total > 0 
                  ? `Processing ${total.toLocaleString()} records...` 
                  : 'Preparing upload...'}
            </h3>
            <div className="text-3xl font-bold text-blue-600 flex items-center justify-center gap-1">
              <AnimatedNumber value={processed} />
              <span>/</span>
              <AnimatedNumber value={total} />
            </div>
          </div>

          {/* Enhanced Stats Grid */}
          <div className="grid grid-cols-2 gap-4">
            {/* Processing Speed */}
            {(currentSpeed > 0 || processingStats.averageSpeed > 0) && (
              <div className="bg-white/50 backdrop-blur-sm rounded-lg p-3 border border-blue-100/50">
                <div className="flex items-center gap-2 text-blue-600 mb-1">
                  <Activity className="h-4 w-4" />
                  <span className="text-xs font-medium">Processing Speed</span>
                </div>
                <div className="text-sm font-semibold text-slate-700">
                  {currentSpeed > 0 ? formatSpeed(currentSpeed) : formatSpeed(processingStats.averageSpeed)}
                </div>
              </div>
            )}

            {/* Time Remaining */}
            {timeRemaining > 0 && !stage.toLowerCase().includes('analysis complete') && (
              <div className="bg-white/50 backdrop-blur-sm rounded-lg p-3 border border-blue-100/50">
                <div className="flex items-center gap-2 text-blue-600 mb-1">
                  <Clock className="h-4 w-4" />
                  <span className="text-xs font-medium">Time Remaining</span>
                </div>
                <div className="text-sm font-semibold text-slate-700">
                  {formatTimeRemaining(timeRemaining)}
                </div>
              </div>
            )}
          </div>

          {/* Progress Stages */}
          <ScrollArea className="h-[200px] rounded-lg border border-blue-100/50 p-4 bg-white/50 backdrop-blur-sm">
            <div className="space-y-2">
              {/* Loading Stage */}
              <div className={`flex items-center gap-2 p-2 rounded-lg transition-colors
                ${isLoading ? 'bg-blue-50 text-blue-700' : 'bg-slate-50 text-slate-500'}`}>
                <FileText className="h-4 w-4" />
                <span className="text-sm">Loading File</span>
                {isLoading && <Loader2 className="h-4 w-4 ml-auto animate-spin" />}
                {!isLoading && <ChevronRight className="h-4 w-4 ml-auto" />}
              </div>

              {/* Processing Stage */}
              <div className={`flex items-center gap-2 p-2 rounded-lg transition-colors
                ${isProcessing ? 'bg-blue-50 text-blue-700' : 'bg-slate-50 text-slate-500'}`}>
                <Database className="h-4 w-4" />
                <span className="text-sm">
                  {isProcessing ? `Processing record ${processed.toLocaleString()} of ${total.toLocaleString()}` : "Processing Records"}
                </span>
                {isProcessing && <Loader2 className="h-4 w-4 ml-auto animate-spin" />}
                {!isProcessing && <ChevronRight className="h-4 w-4 ml-auto" />}
              </div>

              {/* Status Message */}
              <div className="p-2 rounded-lg bg-slate-50">
                <div className="text-sm font-mono text-slate-700 whitespace-pre-wrap break-words">
                  {stage}
                </div>
              </div>
            </div>
          </ScrollArea>

          {/* Enhanced Progress Bar */}
          <div>
            <div className="flex justify-between text-sm font-medium mb-2">
              <span className="text-slate-700">Overall Progress</span>
              <span className={`transition-colors ${stage.toLowerCase().includes('analysis complete') ? 'text-green-600' : 'text-blue-600'}`}>
                {percentComplete}%
              </span>
            </div>
            <div className="h-3 bg-slate-100 rounded-full overflow-hidden shadow-inner">
              <motion.div 
                className={`h-full transition-colors duration-300 ${
                  stage.toLowerCase().includes('analysis complete')
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
                  animation: stage.toLowerCase().includes('analysis complete')
                    ? 'completion-pulse 2s ease-in-out infinite' 
                    : 'progress-flow 2s linear infinite'
                }}
              />
            </div>
          </div>

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

              .bg-grid-slate-200\/50 {
                background-size: 30px 30px;
                background-image: linear-gradient(to right, rgba(148, 163, 184, 0.05) 1px, transparent 1px),
                                 linear-gradient(to bottom, rgba(148, 163, 184, 0.05) 1px, transparent 1px);
              }
            `}
          </style>
        </div>
      </motion.div>
    </motion.div>,
    document.body
  );
}