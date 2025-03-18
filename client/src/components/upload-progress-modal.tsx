import { AnimatePresence, motion } from "framer-motion";
import { Loader2, FileText, Database, ChevronRight } from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { createPortal } from "react-dom";
import { ScrollArea } from "@/components/ui/scroll-area";

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

export function UploadProgressModal() {
  const { isUploading, uploadProgress } = useDisasterContext();

  if (!isUploading) return null;

  const { 
    processed = 0, 
    total = 0, 
    stage = 'Processing...', 
    batchNumber = 0, 
    totalBatches = 0, 
    currentSpeed = 0, 
    timeRemaining = 0, 
    processingStats 
  } = uploadProgress;

  const percentage = total > 0 ? Math.round((processed / total) * 100) : 0;

  // Determine current stage
  const isLoading = stage.toLowerCase().includes('loading') || stage.toLowerCase().includes('initializing');
  const isProcessing = stage.toLowerCase().includes('processing');
  const isCompleted = stage.toLowerCase().includes('complete');

  return createPortal(
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="fixed inset-0 flex items-center justify-center z-[9999]"
    >
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        className="absolute inset-0 bg-black/20 backdrop-blur-sm"
      />

      {/* Content */}
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="relative bg-white/95 backdrop-blur-lg rounded-xl border border-blue-100 p-6 max-w-md w-full mx-4 shadow-2xl"
      >
        {/* Main Progress Display */}
        <div className="text-center mb-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-1">
            {stage}
          </h3>
          <div className="text-3xl font-bold text-blue-600 flex items-center justify-center gap-1">
            <AnimatedNumber value={processed} />
            <span>/</span>
            <AnimatedNumber value={total} />
          </div>
          {currentSpeed > 0 && (
            <div className="text-sm text-slate-600 mt-2">
              Processing {Math.round(currentSpeed)} records/second
              {timeRemaining > 0 && ` • ${Math.round(timeRemaining)}s remaining`}
            </div>
          )}
        </div>

        {/* Detailed Progress Log */}
        <ScrollArea className="h-[200px] rounded-md border p-4">
          <div className="space-y-2">
            {/* Loading Stage */}
            <div className={`flex items-center gap-2 p-2 rounded-lg transition-colors
              ${isLoading ? 'bg-blue-50 text-blue-700' : 'bg-gray-50 text-gray-500'}`}>
              <FileText className="h-4 w-4" />
              <span className="text-sm">Loading File</span>
              {isLoading && <Loader2 className="h-4 w-4 ml-auto animate-spin" />}
              {!isLoading && <ChevronRight className="h-4 w-4 ml-auto" />}
            </div>

            {/* Processing Stage */}
            <div className={`flex items-center gap-2 p-2 rounded-lg transition-colors
              ${isProcessing ? 'bg-blue-50 text-blue-700' : 'bg-gray-50 text-gray-500'}`}>
              <Database className="h-4 w-4" />
              <span className="text-sm">
                {isProcessing && totalBatches > 0 ? 
                  `Processing batch ${batchNumber} of ${totalBatches}` : 
                  'Processing Data'}
              </span>
              {isProcessing && <Loader2 className="h-4 w-4 ml-auto animate-spin" />}
              {!isProcessing && <ChevronRight className="h-4 w-4 ml-auto" />}
            </div>

            {/* Stats */}
            {processingStats && (
              <div className="p-2 rounded-lg bg-gray-50">
                <div className="text-sm text-gray-600">
                  <div>Successful: {processingStats.successCount}</div>
                  {processingStats.errorCount > 0 && (
                    <div className="text-red-600">Errors: {processingStats.errorCount}</div>
                  )}
                  <div>Average Speed: {Math.round(processingStats.averageSpeed)} records/s</div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Progress Bar */}
        <div className="mt-6">
          <div className="flex justify-between text-sm text-slate-600 mb-1">
            <span>Overall Progress</span>
            <span className="font-semibold">{percentage}%</span>
          </div>
          <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-blue-500"
              initial={{ width: 0 }}
              animate={{ width: `${percentage}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>
      </motion.div>
    </motion.div>,
    document.body
  );
}