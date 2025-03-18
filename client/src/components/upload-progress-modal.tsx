import { useState, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";
import { useDisasterContext } from "@/context/disaster-context";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

const AnimatedNumber = ({ value }: { value: number }) => (
  <motion.span
    key={value}
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -20 }}
    transition={{ duration: 0.2 }}
    className="font-mono"
  >
    {value.toLocaleString()}
  </motion.span>
);

export function UploadProgressModal() {
  const { isUploading, uploadProgress } = useDisasterContext();
  const [highestProcessed, setHighestProcessed] = useState(0);

  useEffect(() => {
    if (uploadProgress.processed > 0 && uploadProgress.processed > highestProcessed) {
      setHighestProcessed(uploadProgress.processed);
    }
  }, [uploadProgress.processed, highestProcessed]);

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
    }
  } = uploadProgress;

  const processed = Math.max(rawProcessed, highestProcessed);
  const progress = (processed / total) * 100;
  const isComplete = processed === total;

  return createPortal(
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 flex items-center justify-center z-[9999] p-4"
    >
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
      />

      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className={cn(
          "relative max-w-md w-full p-8 rounded-2xl shadow-2xl",
          "bg-gradient-to-br from-white/95 to-white/90 backdrop-blur-xl",
          "border border-white/20"
        )}
      >
        <div className="space-y-6">
          <div className="text-center">
            <motion.h3
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600"
            >
              {isComplete ? "Analysis Complete!" : "Analyzing Data"}
            </motion.h3>

            <motion.div 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              className="mt-2 text-gray-600"
            >
              {total > 0 ? `Processing ${total} records` : 'Preparing analysis...'}
            </motion.div>
          </div>

          <div className="relative pt-4">
            <Progress 
              value={progress} 
              className={cn(
                "h-3 transition-all duration-500",
                isComplete ? "bg-green-100 [&>[role=progressbar]]:bg-green-500" : 
                "bg-blue-100 [&>[role=progressbar]]:bg-blue-500"
              )}
            />

            <motion.div 
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-2 bg-white rounded-full shadow-lg px-3 py-1 text-sm font-medium"
            >
              {Math.round(progress)}%
            </motion.div>
          </div>

          <div className="grid grid-cols-2 gap-4 text-center">
            <div className="bg-white/50 backdrop-blur-sm rounded-xl p-3 border border-white/20">
              <div className="text-sm text-gray-500">Processed</div>
              <div className="text-2xl font-bold text-blue-600">
                <AnimatedNumber value={processed} />
              </div>
            </div>
            <div className="bg-white/50 backdrop-blur-sm rounded-xl p-3 border border-white/20">
              <div className="text-sm text-gray-500">Total</div>
              <div className="text-2xl font-bold text-blue-600">
                <AnimatedNumber value={total} />
              </div>
            </div>
          </div>

          <div className="text-center text-sm text-gray-500">
            {stage}
          </div>
        </div>
      </motion.div>
    </motion.div>,
    document.body
  );
}