import { AnimatePresence, motion } from "framer-motion";
import { Loader2, CheckCircle, AlertCircle, FileText, Database, ChevronRight } from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { createPortal } from "react-dom";

// Animated number component for smooth transitions
const AnimatedNumber = ({ value }: { value: number }) => (
  <motion.span
    key={value}
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -20 }}
    transition={{ duration: 0.2 }}
  >
    {value}
  </motion.span>
);

// Progress step component
const ProgressStep = ({ 
  icon: Icon, 
  title, 
  isActive, 
  isComplete 
}: { 
  icon: any;
  title: string;
  isActive: boolean;
  isComplete: boolean;
}) => (
  <div className={`
    flex items-center gap-2 p-2 rounded-lg transition-colors
    ${isActive ? 'bg-blue-50' : isComplete ? 'bg-emerald-50' : 'bg-slate-50'}
  `}>
    <div className={`
      p-1.5 rounded-full
      ${isActive ? 'bg-blue-100 text-blue-600' : 
        isComplete ? 'bg-emerald-100 text-emerald-600' : 
        'bg-slate-100 text-slate-400'}
    `}>
      <Icon className="h-4 w-4" />
    </div>
    <span className={`
      text-sm font-medium
      ${isActive ? 'text-blue-700' : 
        isComplete ? 'text-emerald-700' : 
        'text-slate-400'}
    `}>
      {title}
    </span>
    <ChevronRight className={`
      h-4 w-4 ml-auto
      ${isActive ? 'text-blue-400' : 
        isComplete ? 'text-emerald-400' : 
        'text-slate-300'}
    `} />
  </div>
);

export function UploadProgressModal() {
  const { isUploading, uploadProgress } = useDisasterContext();

  // Calculate progress and determine current stage
  const percentage = uploadProgress.totalRecords > 0 
    ? Math.round((uploadProgress.processedRecords / uploadProgress.totalRecords) * 100)
    : 0;

  const stage = uploadProgress.stage || '';
  const isLoading = stage.toLowerCase().includes('loading');
  const isProcessing = stage.toLowerCase().includes('processing');
  const isCompleted = uploadProgress.status === 'success';

  // Enhanced processing message
  const processingMessage = uploadProgress.totalRecords > 0
    ? `Processing record ${uploadProgress.processedRecords} of ${uploadProgress.totalRecords}`
    : uploadProgress.message;

  return createPortal(
    <AnimatePresence mode="wait">
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
            transition={{ 
              duration: 0.2,
              scale: {
                type: "spring",
                damping: 25,
                stiffness: 400
              }
            }}
            className="relative bg-white/95 backdrop-blur-lg rounded-xl border border-blue-100 p-6 max-w-md w-full mx-4 shadow-2xl"
          >
            {/* Progress Steps */}
            <div className="space-y-2 mb-6">
              <ProgressStep
                icon={FileText}
                title="Loading File"
                isActive={isLoading}
                isComplete={isProcessing || isCompleted}
              />
              <ProgressStep
                icon={Database}
                title="Processing Data"
                isActive={isProcessing}
                isComplete={isCompleted}
              />
              <ProgressStep
                icon={CheckCircle}
                title="Analysis Complete"
                isActive={isCompleted}
                isComplete={isCompleted}
              />
            </div>

            {/* Current Status */}
            <div className="flex items-center mb-4">
              {uploadProgress.status === "uploading" && (
                <Loader2 className="animate-spin h-5 w-5 mr-2 text-blue-600" />
              )}
              {uploadProgress.status === "success" && (
                <CheckCircle className="h-5 w-5 mr-2 text-emerald-600" />
              )}
              {uploadProgress.status === "error" && (
                <AlertCircle className="h-5 w-5 mr-2 text-red-600" />
              )}
              <span
                className={`font-medium text-sm ${
                  uploadProgress.status === "error"
                    ? "text-red-800"
                    : uploadProgress.status === "success"
                      ? "text-emerald-800"
                      : "text-blue-800"
                }`}
              >
                {processingMessage}
              </span>
            </div>

            {/* Progress Bar */}
            <div className="relative">
              <div className="overflow-hidden h-2.5 text-xs flex rounded-full bg-slate-200/50 backdrop-blur-sm">
                <motion.div
                  className={`
                    shadow-sm flex flex-col text-center whitespace-nowrap text-white justify-center
                    ${
                      uploadProgress.status === "error"
                        ? "bg-red-500"
                        : uploadProgress.status === "success"
                          ? "bg-emerald-500"
                          : "bg-blue-500"
                    }
                  `}
                  initial={{ width: "0%" }}
                  animate={{ width: `${percentage}%` }}
                  transition={{ 
                    duration: 0.3,
                    ease: "easeInOut"
                  }}
                />
              </div>
            </div>

            {/* Progress Numbers */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ 
                duration: 0.2,
                delay: 0.1
              }}
              className="mt-4 flex flex-col items-center justify-center"
            >
              <div className="text-4xl font-bold text-blue-700 mb-2 tracking-tight tabular-nums flex items-center gap-1">
                <AnimatePresence mode="wait">
                  <AnimatedNumber value={uploadProgress.processedRecords} />
                </AnimatePresence>
                <span>/</span>
                <AnimatePresence mode="wait">
                  <AnimatedNumber value={uploadProgress.totalRecords} />
                </AnimatePresence>
              </div>

              <div className="flex justify-between w-full text-sm text-slate-700">
                <span>Records processed</span>
                <span className="font-semibold tabular-nums">
                  <AnimatePresence mode="wait">
                    <AnimatedNumber value={percentage} />
                  </AnimatePresence>
                  %
                </span>
              </div>
            </motion.div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
}