import { useDisasterContext } from "@/context/disaster-context";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";

export function SimpleProgress() {
  const { uploadProgress } = useDisasterContext();
  
  // Simple percentage calculation
  const percentage = uploadProgress.total ? 
    Math.round((uploadProgress.processed / uploadProgress.total) * 100) : 0;

  if (!uploadProgress.stage) return null;

  return (
    <div className="fixed inset-0 bg-black/20 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl p-6 w-[400px] space-y-4">
        {/* Processing Status */}
        <div className="flex items-center space-x-3">
          <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
          <span className="text-sm font-medium">{uploadProgress.stage}</span>
        </div>

        {/* Numbers */}
        <div className="text-center">
          <span className="text-2xl font-bold tabular-nums">
            {uploadProgress.processed} / {uploadProgress.total}
          </span>
        </div>

        {/* Progress Bar */}
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-blue-600"
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 0.2 }}
          />
        </div>

        {/* Percentage */}
        <div className="text-center text-sm text-gray-600">
          {percentage}% Complete
        </div>
      </div>
    </div>
  );
}
