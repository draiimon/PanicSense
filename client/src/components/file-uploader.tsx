import { useRef, useEffect } from 'react';
import { uploadCSV } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
import { queryClient } from '@/lib/queryClient';
import { useDisasterContext } from '@/context/disaster-context';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Loader2, CheckCircle, AlertCircle } from 'lucide-react';

interface FileUploaderProps {
  onSuccess?: (data: any) => void;
  className?: string;
}

export function FileUploader({ onSuccess, className }: FileUploaderProps) {
  const { toast } = useToast();
  const { 
    isUploading, 
    setIsUploading,
    uploadProgress,
    updateUploadProgress,
    resetUploadProgress
  } = useDisasterContext();
  const progressTimeout = useRef<NodeJS.Timeout>();

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (progressTimeout.current) {
        clearTimeout(progressTimeout.current);
      }
    };
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    if (!file.name.toLowerCase().endsWith('.csv')) {
      toast({
        title: 'Invalid file type',
        description: 'Please upload a CSV file',
        variant: 'destructive',
      });
      return;
    }

    setIsUploading(true);
    updateUploadProgress({
      status: 'uploading',
      message: 'Reading file...',
      percentage: 0,
      processedRecords: 0,
      totalRecords: 0
    });

    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const text = e.target?.result as string;
        const lines = text.split('\n').length - 1;
        updateUploadProgress({ totalRecords: lines });

        try {
          const result = await uploadCSV(file, (progress) => {
            updateUploadProgress({
              processedRecords: progress.processed,
              percentage: Math.min(Math.round((progress.processed / lines) * 100), 100),
              message: `Processing records (${progress.processed}/${lines})`
            });

            if (progress.error) {
              throw new Error(progress.error);
            }
          });

          updateUploadProgress({
            status: 'success',
            message: 'Analysis Complete',
            percentage: 100
          });

          toast({
            title: 'Success!',
            description: `Analyzed ${result.posts.length} posts successfully`,
          });

          // Invalidate queries to refresh data
          queryClient.invalidateQueries({ queryKey: ['/api/sentiment-posts'] });
          queryClient.invalidateQueries({ queryKey: ['/api/analyzed-files'] });
          queryClient.invalidateQueries({ queryKey: ['/api/disaster-events'] });

          if (onSuccess) {
            onSuccess(result);
          }

          // Delay resetting the upload state
          progressTimeout.current = setTimeout(() => {
            resetUploadProgress();
          }, 2000);

        } catch (error) {
          updateUploadProgress({
            status: 'error',
            message: error instanceof Error ? error.message : 'Upload failed'
          });
          throw error;
        }
      };

      reader.readAsText(file);
    } catch (error) {
      updateUploadProgress({
        status: 'error',
        message: error instanceof Error ? error.message : 'Upload failed'
      });

      toast({
        title: 'Upload failed',
        description: error instanceof Error ? error.message : 'An unexpected error occurred',
        variant: 'destructive',
      });

      // Reset state after error
      progressTimeout.current = setTimeout(() => {
        resetUploadProgress();
      }, 2000);
    } finally {
      event.target.value = '';
    }
  };

  return (
    <div className={className}>
      <AnimatePresence mode="wait">
        {uploadProgress.status === 'idle' ? (
          <motion.label
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="inline-flex items-center justify-center w-[140px] h-[40px] bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white text-sm font-medium rounded-md cursor-pointer transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
          >
            <Upload className="h-5 w-5 mr-2" />
            Upload CSV
            <input 
              type="file" 
              className="hidden" 
              accept=".csv" 
              onChange={handleFileUpload}
              disabled={isUploading}
            />
          </motion.label>
        ) : (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex flex-col items-center p-4 bg-white border rounded-lg shadow-lg min-w-[300px]"
          >
            <div className="flex items-center mb-3">
              {uploadProgress.status === 'uploading' && (
                <Loader2 className="animate-spin h-5 w-5 mr-3 text-blue-600" />
              )}
              {uploadProgress.status === 'success' && (
                <CheckCircle className="h-5 w-5 mr-3 text-green-600" />
              )}
              {uploadProgress.status === 'error' && (
                <AlertCircle className="h-5 w-5 mr-3 text-red-600" />
              )}
              <span className={`font-medium ${
                uploadProgress.status === 'error' ? 'text-red-800' :
                uploadProgress.status === 'success' ? 'text-green-800' :
                'text-blue-800'
              }`}>
                {uploadProgress.message}
              </span>
            </div>

            <motion.div 
              className="w-full bg-gray-200 rounded-full h-2.5 mb-2 overflow-hidden"
            >
              <motion.div 
                className={`h-2.5 rounded-full ${
                  uploadProgress.status === 'error' ? 'bg-red-600' :
                  uploadProgress.status === 'success' ? 'bg-green-600' :
                  'bg-blue-600'
                }`}
                initial={{ width: 0 }}
                animate={{ width: `${uploadProgress.percentage}%` }}
                transition={{ duration: 0.3 }}
              />
            </motion.div>

            {uploadProgress.totalRecords > 0 && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-xs text-gray-500 text-center"
              >
                Processed {uploadProgress.processedRecords} of {uploadProgress.totalRecords} records
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}