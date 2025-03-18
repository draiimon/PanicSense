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
    {value}
  </motion.span>
);

export function UploadProgressModal() {
  const { isUploading, uploadProgress } = useDisasterContext();
  console.log('Progress Modal State:', { isUploading, uploadProgress }); // Debug log

  // Calculate current stage and progress
  const { processed, total, stage } = uploadProgress;
  const percentage = total > 0 ? Math.round((processed / total) * 100) : 0;

  // Determine stages based on stage string
  const isLoading = stage.toLowerCase().includes('loading');
  const isProcessing = stage.toLowerCase().includes('processing');
  const isCompleted = stage.toLowerCase().includes('complete');

  // Enhanced stage message
  const getStageMessage = () => {
    if (isCompleted) return 'Analysis Complete!';
    if (isProcessing) return `Processing record ${processed} of ${total}`;
    if (isLoading) return stage;
    return 'Preparing...';
  };

  return createPortal(
    <AnimatePresence>
      {isUploading && (
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
                {getStageMessage()}
              </h3>
              <div className="text-3xl font-bold text-blue-600 flex items-center justify-center gap-1">
                <AnimatedNumber value={processed} />
                <span>/</span>
                <AnimatedNumber value={total} />
              </div>
            </div>

            {/* Scrollable Progress Log */}
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
                  <span className="text-sm">Analyzing Data</span>
                  {isProcessing && <Loader2 className="h-4 w-4 ml-auto animate-spin" />}
                  {!isProcessing && <ChevronRight className="h-4 w-4 ml-auto" />}
                </div>

                {/* Current Progress */}
                <div className="p-2 rounded-lg bg-gray-50">
                  <span className="text-sm text-gray-600">Current Stage: {stage}</span>
                </div>
              </div>
            </ScrollArea>

            {/* Progress Bar */}
            <div className="mt-6">
              <div className="flex justify-between text-sm text-slate-600 mb-1">
                <span>Overall Progress</span>
                <span className="font-semibold">
                  {percentage}%
                </span>
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
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
}