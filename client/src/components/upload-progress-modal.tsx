import { AnimatePresence, motion } from "framer-motion";
import { Loader2, FileText, Database, ChevronRight } from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { createPortal } from "react-dom";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useEffect, useState, useRef } from "react";
// import { getPythonConsoleMessages, PythonConsoleMessage } from "@/lib/api"; //Removed import
// import { useQuery } from "@tanstack/react-query"; //Removed import

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
  const [activeTab, setActiveTab] = useState<'progress'>('progress');
  const consoleScrollRef = useRef<HTMLDivElement>(null);
  const [highestProcessed, setHighestProcessed] = useState(0);

  // Style to center the modal content
  const centerStyle = {
    position: 'fixed' as const,
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: '90%',
    maxWidth: '500px',
    backgroundColor: 'white',
    padding: '2rem',
    borderRadius: '0.5rem',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    zIndex: 50
  };

  // Removed useQuery hook and related code

  // Removed useEffect for auto-scrolling console

  // Effect to track the highest processed value to prevent jumping backward
  useEffect(() => {
    // Only update the highest processed value if it's greater than current highest
    // AND only if we're not at the very beginning (processed > 0)
    if (uploadProgress.processed > 0 && uploadProgress.processed > highestProcessed) {
      setHighestProcessed(uploadProgress.processed);
      console.log(`Updated highest processed value to ${uploadProgress.processed}`);
    }
  }, [uploadProgress.processed, highestProcessed]);

  // Reset highest processed value when modal is closed/hidden
  useEffect(() => {
    if (!isUploading) {
      // When the dialog closes, reset our tracking to ensure fresh start next time
      setHighestProcessed(0);
      console.log('Upload Progress Modal closed - reset highest processed value');
    }
  }, [isUploading]);

  if (!isUploading) return null;

  return (
    <div style={centerStyle} className="space-y-4">

  const { 
    stage = 'Processing...', 
    processed: rawProcessed = 0, 
    total = 100,
    processingStats = {
      successCount: 0,
      errorCount: 0,
      averageSpeed: 0
    },
    currentSpeed = 0
  } = uploadProgress;

  // Use the higher value between current processed and highest recorded
  // This prevents the counter from going backward
  const processed = Math.max(rawProcessed, highestProcessed);

  // Stage indication
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
            {total > 0 ? `Initializing analysis for ${total} records...` : 'Preparing upload...'}
          </h3>
          <div className="text-3xl font-bold text-blue-600 flex items-center justify-center gap-1">
            <AnimatedNumber value={processed} />
            <span>/</span>
            <AnimatedNumber value={total} />
          </div>
        </div>

        {/* Tab Navigation - Console tab removed */}
        <div className="flex mb-4 border-b">
          <button 
            onClick={() => setActiveTab('progress')} 
            className={`flex items-center gap-1 px-4 py-2 border-b-2 transition-colors ${activeTab === 'progress' 
              ? 'border-blue-500 text-blue-600' 
              : 'border-transparent text-gray-500 hover:text-gray-700'}`}
          >
            <Database className="h-4 w-4" />
            <span>Progress</span>
          </button>
        </div>

        {activeTab === 'progress' && (
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
                  {isProcessing && `Processing record ${processed} of ${total}`}
                  {!isProcessing && "Processing Records"}
                </span>
                {isProcessing && <Loader2 className="h-4 w-4 ml-auto animate-spin" />}
                {!isProcessing && <ChevronRight className="h-4 w-4 ml-auto" />}
              </div>

              {/* Console Message */}
              <div className="p-2 rounded-lg bg-gray-50">
                <div className="text-sm font-mono text-gray-700 whitespace-pre-wrap overflow-x-auto">
                  {stage.includes("PROGRESS:") ? stage : stage.includes("Completed record") ? stage : "Processing..."}
                </div>
              </div>


            </div>
          </ScrollArea>
        )}


        {/* Enhanced Progress Bar */}
        <div className="mt-6">
          <div className="flex justify-between text-sm font-medium mb-2">
            <span className="text-slate-700">Overall Progress</span>
            <span className={`transition-colors ${processed === total ? 'text-green-600' : 'text-blue-600'}`}>
              {Math.round((processed / total) * 100)}%
            </span>
          </div>
          <div className="h-3 bg-slate-100 rounded-full overflow-hidden shadow-inner">
            <motion.div 
              className={`h-full transition-colors duration-300 ${
                processed === total 
                  ? 'bg-gradient-to-r from-green-400 to-green-500' 
                  : 'bg-gradient-to-r from-blue-400 via-blue-500 to-blue-400'
              }`}
              initial={{ width: 0 }}
              animate={{ 
                width: `${(processed / total) * 100}%`,
                transition: { duration: 0.3 }
              }}
              style={{
                backgroundSize: '200% 100%',
                animation: processed === total 
                  ? 'completion-pulse 1s ease-in-out' 
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
                opacity: 1;
                transform: scale(1);
              }
            }
          `}
        </style>
      </motion.div>
    </motion.div>,
    document.body
  );
}