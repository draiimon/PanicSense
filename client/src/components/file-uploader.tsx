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
        title: 'Invalid File Format',
        description: 'Please upload a CSV file containing disaster-related data.',
        variant: 'destructive',
      });
      return;
    }

    setIsUploading(true);
    updateUploadProgress({
      status: 'uploading',
      message: 'Initializing analysis...',
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
              message: `Analyzing sentiment data... ${progress.processed}/${lines}`
            });

            if (progress.error) {
              throw new Error(progress.error);
            }
          });

          updateUploadProgress({
            status: 'success',
            message: 'Analysis Complete!',
            percentage: 100
          });

          toast({
            title: '🎉 Analysis Complete',
            description: `Successfully analyzed ${result.posts.length} posts with sentiment data`,
            duration: 5000,
          });

          // Invalidate queries to refresh data
          queryClient.invalidateQueries({ queryKey: ['/api/sentiment-posts'] });
          queryClient.invalidateQueries({ queryKey: ['/api/analyzed-files'] });
          queryClient.invalidateQueries({ queryKey: ['/api/disaster-events'] });

          if (onSuccess) {
            onSuccess(result);
          }

          // Delay resetting upload state
          progressTimeout.current = setTimeout(() => {
            resetUploadProgress();
          }, 2000);

        } catch (error) {
          handleError(error);
        }
      };

      reader.readAsText(file);
    } catch (error) {
      handleError(error);
    } finally {
      event.target.value = '';
    }
  };

  const handleError = (error: unknown) => {
    updateUploadProgress({
      status: 'error',
      message: error instanceof Error ? error.message : 'Upload failed'
    });

    toast({
      title: 'Analysis Failed',
      description: error instanceof Error ? error.message : 'An unexpected error occurred during analysis',
      variant: 'destructive',
    });

    progressTimeout.current = setTimeout(() => {
      resetUploadProgress();
    }, 2000);
  };

  return (
    <>
      {/* Upload Button */}
      <motion.label
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`
          inline-flex items-center justify-center px-6 py-3
          bg-gradient-to-r from-blue-600 to-indigo-600
          hover:from-blue-700 hover:to-indigo-700
          text-white text-sm font-medium rounded-full
          cursor-pointer transition-all duration-300
          shadow-lg hover:shadow-xl transform hover:-translate-y-0.5
          disabled:opacity-50 disabled:cursor-not-allowed
          ${className}
        `}
      >
        <Upload className="h-5 w-5 mr-2" />
        Upload Dataset
        <input 
          type="file" 
          className="hidden" 
          accept=".csv" 
          onChange={handleFileUpload}
          disabled={isUploading}
        />
      </motion.label>

      {/* Fixed Progress Overlay */}
      <AnimatePresence>
        {isUploading && (
          <motion.div 
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed bottom-6 right-6 z-50"
          >
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl border border-slate-200/50 p-6 max-w-md">
              <div className="flex items-center mb-4">
                {uploadProgress.status === 'uploading' && (
                  <Loader2 className="animate-spin h-6 w-6 mr-3 text-blue-600" />
                )}
                {uploadProgress.status === 'success' && (
                  <CheckCircle className="h-6 w-6 mr-3 text-emerald-600" />
                )}
                {uploadProgress.status === 'error' && (
                  <AlertCircle className="h-6 w-6 mr-3 text-red-600" />
                )}
                <span className={`font-medium text-lg ${
                  uploadProgress.status === 'error' ? 'text-red-800' :
                  uploadProgress.status === 'success' ? 'text-emerald-800' :
                  'text-blue-800'
                }`}>
                  {uploadProgress.message}
                </span>
              </div>

              <div className="relative">
                <div className="overflow-hidden h-2 text-xs flex rounded-full bg-slate-200/50 backdrop-blur-sm">
                  <motion.div
                    className={`
                      shadow-lg flex flex-col text-center whitespace-nowrap text-white justify-center
                      ${uploadProgress.status === 'error' ? 'bg-red-500' :
                        uploadProgress.status === 'success' ? 'bg-emerald-500' :
                        'bg-blue-500'
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
                  className="mt-3 text-sm text-slate-600 flex justify-between items-center"
                >
                  <span>
                    Processing: {uploadProgress.processedRecords} of {uploadProgress.totalRecords}
                  </span>
                  <span className="font-semibold">
                    {uploadProgress.percentage}%
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