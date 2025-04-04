import { useState, useEffect } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { Loader2, CheckCircle, AlertCircle, Upload, AlertTriangle } from "lucide-react";
import { FileUploaderButton } from "./file-uploader-button";
import { motion } from "framer-motion";
import { Progress } from "@/components/ui/progress";

interface FileUploaderProps {
  onSuccess?: (data: any) => void;
  className?: string;
}

export function FileUploader({ onSuccess, className }: FileUploaderProps) {
  const { isUploading, uploadProgress } = useDisasterContext();
  const [showProgress, setShowProgress] = useState(false);
  
  // Show progress bar with a slight delay for better UX
  useEffect(() => {
    let timer: NodeJS.Timeout;
    
    if (isUploading) {
      timer = setTimeout(() => {
        setShowProgress(true);
      }, 300);
    } else {
      // Hide progress with a delay to allow final status to be seen
      timer = setTimeout(() => {
        setShowProgress(false);
      }, 1500);
    }
    
    return () => clearTimeout(timer);
  }, [isUploading]);
  
  // Calculate progress percentage
  const progressPercentage = uploadProgress.total > 0 
    ? Math.round((uploadProgress.processed / uploadProgress.total) * 100)
    : 0;
    
  // Determine status icon
  const getStatusIcon = () => {
    if (uploadProgress.stage?.includes("error") || uploadProgress.stage?.includes("failed")) {
      return <AlertTriangle className="h-4 w-4 text-red-500" />;
    } else if (uploadProgress.stage?.includes("complete") || 
        uploadProgress.stage?.includes("finished") || 
        progressPercentage >= 100) {
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    } else {
      return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
    }
  };
  
  return (
    <div className="w-full flex flex-col gap-2">
      <FileUploaderButton onSuccess={onSuccess} className={className} />
      
      {/* Simple inline progress indicator */}
      {(isUploading || showProgress) && (
        <motion.div 
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.3 }}
          className="mt-2 bg-white/80 backdrop-blur-sm p-3 rounded-lg border border-gray-200 shadow-sm"
        >
          <div className="flex items-center gap-2 mb-1.5">
            {getStatusIcon()}
            <p className="text-sm font-medium text-gray-700 truncate">
              {uploadProgress.stage || 'Processing...'}
            </p>
          </div>
          
          <Progress value={progressPercentage} className="h-2" />
          
          <div className="flex justify-between mt-1.5 text-xs text-gray-500">
            <span>{uploadProgress.processed} of {uploadProgress.total} rows</span>
            <span>{progressPercentage}%</span>
          </div>
        </motion.div>
      )}
    </div>
  );
}