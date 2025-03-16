import { AnimatePresence, motion } from "framer-motion";
import { useDisasterContext } from "@/context/disaster-context";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { FileUploaderButton } from "./file-uploader-button";
import { Progress } from "@/components/ui/progress";

interface FileUploaderProps {
  onSuccess?: (data: any) => void;
  className?: string;
}

export function FileUploader({ onSuccess, className }: FileUploaderProps) {
  const { isUploading, uploadProgress } = useDisasterContext();

  return (
    <>
      {/* Upload Button */}
      <FileUploaderButton onSuccess={onSuccess} className={className} />

      {/* Progress Modal */}
      <AnimatePresence>
        {isUploading && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed inset-0 flex items-center justify-center z-[9999] bg-black/20 backdrop-blur-sm"
          >
            <div className="bg-white/95 backdrop-blur-lg rounded-xl border border-blue-100 p-6 max-w-md w-full mx-4 shadow-2xl">
              {/* Status Header */}
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
                  {uploadProgress.message || "Processing..."}
                </span>
              </div>

              {/* Progress Bar */}
              <Progress 
                value={uploadProgress.percentage || 0} 
                className={`h-2 ${
                  uploadProgress.status === "error"
                    ? "bg-red-200 [&>div]:bg-red-500"
                    : uploadProgress.status === "success"
                      ? "bg-emerald-200 [&>div]:bg-emerald-500"
                      : "bg-blue-200 [&>div]:bg-blue-500"
                }`}
              />

              {/* Progress Details */}
              {uploadProgress.totalRecords > 0 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-2 text-xs text-slate-600 flex justify-between items-center"
                >
                  <span>
                    Processing: {uploadProgress.processedRecords || 0} of{" "}
                    {uploadProgress.totalRecords}
                  </span>
                  <span className="font-semibold">
                    {uploadProgress.percentage || 0}%
                  </span>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}