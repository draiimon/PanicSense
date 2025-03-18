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
    className="tabular-nums"
  >
    {value}
  </motion.span>
);

// Progress step component with animations
const ProgressStep = ({ 
  icon: Icon, 
  title, 
  subtitle,
  isActive, 
  isComplete,
  percentage = 0
}: { 
  icon: any;
  title: string;
  subtitle?: string;
  isActive: boolean;
  isComplete: boolean;
  percentage?: number;
}) => (
  <motion.div 
    className={`
      flex items-center gap-3 p-3 rounded-lg transition-colors
      ${isActive ? 'bg-blue-50/80' : isComplete ? 'bg-emerald-50/80' : 'bg-slate-50/80'}
    `}
    initial={{ opacity: 0, x: -20 }}
    animate={{ opacity: 1, x: 0 }}
    transition={{ duration: 0.3 }}
  >
    <div className={`
      p-2 rounded-full
      ${isActive ? 'bg-blue-100 text-blue-600' : 
        isComplete ? 'bg-emerald-100 text-emerald-600' : 
        'bg-slate-100 text-slate-400'}
    `}>
      <Icon className="h-5 w-5" />
    </div>
    <div className="flex-1">
      <div className="flex items-center justify-between">
        <span className={`
          font-medium
          ${isActive ? 'text-blue-700' : 
            isComplete ? 'text-emerald-700' : 
            'text-slate-500'}
        `}>
          {title}
        </span>
        {percentage > 0 && (
          <span className="text-sm font-medium text-slate-600">
            {percentage}%
          </span>
        )}
      </div>
      {subtitle && (
        <span className="text-sm text-slate-500">
          {subtitle}
        </span>
      )}
      {isActive && percentage > 0 && (
        <div className="mt-2 h-1 bg-slate-200 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-blue-500"
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      )}
    </div>
    <ChevronRight className={`
      h-5 w-5
      ${isActive ? 'text-blue-400' : 
        isComplete ? 'text-emerald-400' : 
        'text-slate-300'}
    `} />
  </motion.div>
);

export function UploadProgressModal() {
  const { isUploading, uploadProgress } = useDisasterContext();

  // Calculate current stage and progress
  const { processed, total, stage } = uploadProgress;
  const percentage = total > 0 ? Math.round((processed / total) * 100) : 0;

  // Determine active stages
  const isLoading = stage.toLowerCase().includes('loading') || stage.toLowerCase().includes('identifying');
  const isProcessing = stage.toLowerCase().includes('processing') || stage.toLowerCase().includes('analyzing');
  const isCompleted = stage.toLowerCase().includes('complete');

  // Get stage-specific progress
  const loadingProgress = isLoading ? Math.min(percentage, 100) : 0;
  const processingProgress = isProcessing ? percentage : 0;

  // Enhanced stage messages
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

            {/* Detailed Progress Steps */}
            <div className="space-y-3">
              <ProgressStep
                icon={FileText}
                title="Loading File"
                subtitle={isLoading ? stage : undefined}
                isActive={isLoading}
                isComplete={isProcessing || isCompleted}
                percentage={loadingProgress}
              />
              <ProgressStep
                icon={Database}
                title="Processing Data"
                subtitle={isProcessing ? `Record ${processed}/${total}` : undefined}
                isActive={isProcessing}
                isComplete={isCompleted}
                percentage={processingProgress}
              />
              <ProgressStep
                icon={CheckCircle}
                title="Analysis Complete"
                isActive={isCompleted}
                isComplete={isCompleted}
              />
            </div>

            {/* Overall Progress */}
            <div className="mt-6">
              <div className="flex justify-between text-sm text-slate-600 mb-1">
                <span>Overall Progress</span>
                <span className="font-semibold">
                  <AnimatedNumber value={percentage} />%
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