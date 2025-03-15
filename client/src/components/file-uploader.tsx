import { AnimatePresence, motion } from "framer-motion";
import { useDisasterContext } from "@/context/disaster-context";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { FileUploaderButton } from "./file-uploader-button";

interface FileUploaderProps {
  onSuccess?: (data: any) => void;
  className?: string;
  containedProgress?: boolean;
}

export function FileUploader({ 
  onSuccess, 
  className,
  containedProgress = false 
}: FileUploaderProps) {
  const { isUploading, uploadProgress } = useDisasterContext();

  return (
    <div className="relative">
      {/* Upload Button */}
      <FileUploaderButton onSuccess={onSuccess} className={className} />

      {/* Progress Indicator */}
      <AnimatePresence>
        {isUploading && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className={`${
              containedProgress 
                ? 'absolute top-full right-0 mt-4' 
                : 'fixed inset-0 flex items-center justify-center'
            } z-[9999] ${!containedProgress && 'bg-black/20 backdrop-blur-sm'}`}
          >
            <div className={`bg-white/95 backdrop-blur-lg rounded-xl border border-blue-100 p-6 shadow-2xl ${
              containedProgress ? 'w-96' : 'max-w-md w-full mx-4'
            }`}>
              {/* Progress Header */}
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
                  {uploadProgress.message}
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
                    initial={{ width: 0 }}
                    animate={{ width: `${uploadProgress.percentage}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
              </div>

              {/* Progress Details */}
              <div className="mt-4 space-y-2">
                {/* Percentage Display */}
                <div className="flex justify-center">
                  <span className="text-2xl font-bold text-slate-800">
                    {uploadProgress.percentage}%
                  </span>
                </div>

                {/* Records Progress */}
                {uploadProgress.totalRecords > 0 && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-sm text-slate-600 flex justify-between items-center"
                  >
                    <span>
                      Records Analyzed: {uploadProgress.processedRecords} of{" "}
                      {uploadProgress.totalRecords}
                    </span>
                  </motion.div>
                )}

                {/* Status Message */}
                <div className="text-sm text-slate-500 text-center">
                  {uploadProgress.stage || "Analyzing data..."}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}