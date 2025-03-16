import { AnimatePresence, motion } from "framer-motion";
import { useDisasterContext } from "@/context/disaster-context";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { FileUploaderButton } from "./file-uploader-button";

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

      {/* Global Progress Indicator - Always rendered from the same place */}
      <AnimatePresence>
        {isUploading && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed inset-0 flex items-center justify-center z-[9999] bg-black/20 backdrop-blur-sm"
          >
            <div className="bg-white/95 backdrop-blur-lg rounded-xl border border-blue-100 p-6 max-w-md w-full mx-4 shadow-2xl">
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

              {uploadProgress.totalRecords > 0 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-4 flex flex-col items-center justify-center"
                >
                  {/* Counter in larger format */}
                  <div className="text-3xl font-bold text-blue-700 mb-2">
                    {uploadProgress.processedRecords}/{uploadProgress.totalRecords}
                  </div>
                  
                  {/* Percentage and label */}
                  <div className="flex justify-between w-full text-sm text-slate-700">
                    <span>
                      Records processed
                    </span>
                    <span className="font-semibold">
                      {uploadProgress.percentage}%
                    </span>
                  </div>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
