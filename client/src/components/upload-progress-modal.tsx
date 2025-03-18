import { AnimatePresence, motion } from "framer-motion";
import { Loader2, FileText, Database, ChevronRight } from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { createPortal } from "react-dom";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useEffect, useState } from "react";

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
  const [simulatedProgress, setSimulatedProgress] = useState({
    processed: 0,
    successCount: 0,
    averageSpeed: 0
  });

  // Simulate progress updates
  useEffect(() => {
    if (!isUploading) {
      setSimulatedProgress({ processed: 0, successCount: 0, averageSpeed: 0 });
      return;
    }

    const interval = setInterval(() => {
      setSimulatedProgress(prev => ({
        processed: Math.min(prev.processed + 1, 20),
        successCount: Math.min(prev.processed + 1, 20),
        averageSpeed: Math.min(prev.averageSpeed + 0.5, 5)
      }));
    }, 300);

    return () => clearInterval(interval);
  }, [isUploading]);

  if (!isUploading) return null;

  const { stage = 'Processing...' } = uploadProgress;

  // Always show animation instead of actual percentage
  const isLoading = stage.toLowerCase().includes('initializing') || stage.toLowerCase().includes('loading');
  const isProcessing = stage.toLowerCase().includes('processing') || stage.toLowerCase().includes('record');
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
            <AnimatedNumber value={simulatedProgress.processed} />
            <span>/</span>
            <AnimatedNumber value={20} />
          </div>
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
                {isProcessing && `Processing record ${simulatedProgress.processed} of 20`}
              </span>
              {isProcessing && <Loader2 className="h-4 w-4 ml-auto animate-spin" />}
              {!isProcessing && <ChevronRight className="h-4 w-4 ml-auto" />}
            </div>

            {/* Stats */}
            <div className="p-2 rounded-lg bg-gray-50">
              <div className="text-sm text-gray-600">
                <div>Successful: {simulatedProgress.successCount}</div>
                <div>Average Speed: {simulatedProgress.averageSpeed.toFixed(1)} records/s</div>
              </div>
            </div>
          </div>
        </ScrollArea>

        {/* Download-style Progress Bar */}
        <div className="mt-6">
          <div className="flex justify-between text-sm text-slate-600 mb-1">
            <span>Overall Progress</span>
          </div>
          <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-blue-500 via-blue-600 to-blue-500" 
                 style={{
                   width: `${(simulatedProgress.processed / 20) * 100}%`,
                   backgroundSize: '200% 100%',
                   animation: 'download-progress 1.5s ease-in-out infinite'
                 }}
            />
          </div>
        </div>

        {/* Add keyframes for smooth download-style animation */}
        <style>
          {`
            @keyframes download-progress {
              0% {
                background-position: 100% 0;
                opacity: 0.5;
              }
              50% {
                background-position: 0 0;
                opacity: 1;
              }
              100% {
                background-position: -100% 0;
                opacity: 0.5;
              }
            }
          `}
        </style>
      </motion.div>
    </motion.div>,
    document.body
  );
}