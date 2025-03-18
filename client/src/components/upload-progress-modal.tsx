import { AnimatePresence, motion } from "framer-motion";
import { Loader2, FileText, Database, ChevronRight, Terminal } from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { createPortal } from "react-dom";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useEffect, useState, useRef } from "react";
import { getPythonConsoleMessages, PythonConsoleMessage } from "@/lib/api";
import { useQuery } from "@tanstack/react-query";

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
  const [activeTab, setActiveTab] = useState<'progress'|'console'>('progress');
  const consoleScrollRef = useRef<HTMLDivElement>(null);
  const [highestProcessed, setHighestProcessed] = useState(0);
  
  // Fetch Python console messages
  const { data: consoleMessages = [], refetch: refetchConsoleMessages } = useQuery({
    queryKey: ['/api/python-console-messages'],
    queryFn: async () => {
      try {
        return await getPythonConsoleMessages(100);
      } catch (error) {
        // Silent error handling - don't log to console
        return [];
      }
    },
    refetchInterval: isUploading ? 500 : false, // Poll every 500ms while uploading
    enabled: isUploading
  });
  
  // Auto-scroll console to bottom when new messages arrive
  useEffect(() => {
    if (consoleScrollRef.current && consoleMessages.length > 0) {
      setTimeout(() => {
        if (consoleScrollRef.current) {
          const scrollElement = consoleScrollRef.current.querySelector('[data-radix-scroll-area-viewport]');
          if (scrollElement) {
            scrollElement.scrollTop = scrollElement.scrollHeight;
          }
        }
      }, 100);
    }
  }, [consoleMessages.length]);

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
            {stage}
          </h3>
          <div className="text-3xl font-bold text-blue-600 flex items-center justify-center gap-1">
            <AnimatedNumber value={processed} />
            <span>/</span>
            <AnimatedNumber value={total} />
          </div>
        </div>

        {/* Tab Navigation */}
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
          <button 
            onClick={() => setActiveTab('console')} 
            className={`flex items-center gap-1 px-4 py-2 border-b-2 transition-colors ${activeTab === 'console' 
              ? 'border-blue-500 text-blue-600' 
              : 'border-transparent text-gray-500 hover:text-gray-700'}`}
          >
            <Terminal className="h-4 w-4" />
            <span>Console</span>
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

              {/* Stats */}
              <div className="p-2 rounded-lg bg-gray-50">
                <div className="text-sm text-gray-600">
                  <div>Successful: {processingStats.successCount}</div>
                  <div>Errors: {processingStats.errorCount}</div>
                  <div>Current Speed: {currentSpeed?.toFixed(1)} records/s</div>
                  <div>Average Speed: {processingStats.averageSpeed?.toFixed(1)} records/s</div>
                </div>
              </div>
            </div>
          </ScrollArea>
        )}

        {activeTab === 'console' && (
          <ScrollArea ref={consoleScrollRef} className="h-[200px] rounded-md border">
            <div className="p-2 bg-black text-green-400 font-mono text-xs space-y-1">
              {consoleMessages.length > 0 ? (
                consoleMessages
                  .filter(msg => {
                    // Filter out technical error messages that don't provide value to users
                    const message = msg.message.toLowerCase();
                    return !(
                      message.includes('traceback') && message.includes('error:') ||
                      message.includes('python process error') ||
                      message.includes('unhandled exception') ||
                      message.includes('command failed with exit code')
                    );
                  })
                  .map((msg, index) => {
                    // Check if it's a progress message and make it stand out
                    const isProgressMsg = msg.message.includes('PROGRESS:');
                    const isSuccessMsg = msg.message.toLowerCase().includes('success') || 
                                         msg.message.toLowerCase().includes('completed');
                    
                    return (
                      <div 
                        key={index} 
                        className={`py-1 ${isProgressMsg ? 'text-blue-400' : ''} ${isSuccessMsg ? 'text-green-400 font-semibold' : ''}`}
                      >
                        <span className="text-gray-500">[{new Date(msg.timestamp).toLocaleTimeString()}]</span>{' '}
                        {/* Clean up PROGRESS: prefix for better readability */}
                        {isProgressMsg 
                          ? msg.message.replace('PROGRESS:', 'Progress:') 
                          : msg.message}
                      </div>
                    );
                  })
              ) : (
                <div className="py-2 text-gray-500 italic">No console output available...</div>
              )}
            </div>
          </ScrollArea>
        )}

        {/* Download-style Progress Bar */}
        <div className="mt-6">
          <div className="flex justify-between text-sm text-slate-600 mb-1">
            <span>Overall Progress</span>
            <span>{Math.round((processed / total) * 100)}%</span>
          </div>
          <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-blue-500 via-blue-600 to-blue-500" 
                 style={{
                   width: `${(processed / total) * 100}%`,
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