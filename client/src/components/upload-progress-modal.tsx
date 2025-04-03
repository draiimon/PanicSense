import { motion } from "framer-motion";
import { CheckCircle, Loader2, XCircle } from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { createPortal } from "react-dom";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { cancelUpload } from "@/lib/api";

export function UploadProgressModal() {
  const { isUploading, uploadProgress, setIsUploading } = useDisasterContext();
  const [highestProcessed, setHighestProcessed] = useState(0);
  const [isCancelling, setIsCancelling] = useState(false);
  const [showCancelDialog, setShowCancelDialog] = useState(false);

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
    total = 100
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
      {/* Simpler backdrop */}
      <div className="absolute inset-0 bg-black/30 backdrop-blur-sm"></div>

      {/* Content */}
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="relative bg-slate-900 rounded-lg overflow-hidden border border-slate-800 w-full max-w-xs mx-4 shadow-xl"
      >
        <div className="p-5">
          {/* Header */}
          <h3 className="text-lg font-semibold text-slate-100 text-center mb-2">
            {isComplete ? 'Analysis Complete' : hasError ? 'Error Occurred' : `Processing ${total} records...`}
          </h3>
          
          {/* Counter */}
          <div className="flex items-center justify-center my-3">
            <motion.span 
              key={processed}
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-3xl font-semibold text-blue-400"
            >
              {processed}
            </motion.span>
            <span className="text-3xl mx-2 text-slate-400">/</span>
            <span className="text-3xl font-semibold text-slate-300">{total}</span>
          </div>
          
          {/* Processing Indicator */}
          <div className="flex items-center justify-center my-4">
            <div className={`flex items-center justify-center w-10 h-10 rounded-full 
              ${isComplete 
                ? 'bg-green-500/10 text-green-400' 
                : hasError 
                  ? 'bg-red-500/10 text-red-400' 
                  : 'bg-blue-500/10 text-blue-400'}`
            }>
              {isComplete ? (
                <CheckCircle className="h-6 w-6" />
              ) : hasError ? (
                <XCircle className="h-6 w-6" />
              ) : (
                <Loader2 className="h-6 w-6 animate-spin" />
              )}
            </div>
          </div>
          
          {/* Progress Bar */}
          <div className="mt-2">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-400">Overall Progress</span>
              <span className={`
                ${isComplete ? 'text-green-400' : hasError ? 'text-red-400' : 'text-blue-400'} font-medium
              `}>
                {percentComplete}%
              </span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <motion.div 
                className={`h-full ${
                  hasError 
                    ? 'bg-red-500'
                    : isComplete
                      ? 'bg-green-500' 
                      : 'bg-blue-500'
                }`}
                initial={{ width: 0 }}
                animate={{ 
                  width: `${percentComplete}%`,
                  transition: { duration: 0.3 }
                }}
              />
            </div>
          </div>
          
          {/* Cancel button */}
          {!isComplete && !hasError && (
            <div className="mt-5 text-center">
              <Button
                variant="destructive"
                size="sm"
                className="gap-1 bg-red-800 hover:bg-red-700"
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
      </motion.div>
      
      {/* Smaller and more compact Cancel Confirmation Dialog */}
      {showCancelDialog && createPortal(
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[10000]" onClick={() => setShowCancelDialog(false)}>
          <div className="bg-slate-900 border border-slate-700 p-4 rounded-lg max-w-xs mx-4" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-semibold text-red-400 mb-2">Cancel Upload?</h3>
            <p className="text-slate-300 text-sm mb-3">
              All progress will be lost. Continue?
            </p>
            <div className="flex justify-end gap-2">
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setShowCancelDialog(false)}
                className="bg-slate-800 text-slate-200 hover:bg-slate-700 hover:text-white border-slate-700"
              >
                No, Continue
              </Button>
              <Button 
                variant="destructive"
                size="sm"
                onClick={handleCancel}
                className="bg-red-800 hover:bg-red-700 text-white border-none"
              >
                Yes, Cancel
              </Button>
            </div>
          </div>
        </div>,
        document.body
      )}
    </motion.div>,
    document.body
  );
}