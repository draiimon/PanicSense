import { useRef } from 'react';
import { Upload } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDisasterContext } from '@/context/disaster-context';
import { useToast } from '@/hooks/use-toast';
import { uploadCSV } from '@/lib/api';
import { queryClient } from '@/lib/queryClient';

interface FileUploaderButtonProps {
  onSuccess?: (data: any) => void;
  className?: string;
}

export function FileUploaderButton({ onSuccess, className }: FileUploaderButtonProps) {
  const { toast } = useToast();
  const { 
    isUploading, 
    setIsUploading,
    uploadProgress,
    updateUploadProgress,
    resetUploadProgress
  } = useDisasterContext();
  const progressTimeout = useRef<NodeJS.Timeout>();

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
            // Normalize the progress data with default values
            const processedRecords = progress.processed || 0;
            const totalRecords = lines || 100;
            const percentage = Math.min(Math.round((processedRecords / totalRecords) * 100), 100);
            
            // Update the progress in the UI with proper default values
            updateUploadProgress({
              processedRecords: processedRecords,
              totalRecords: totalRecords,
              percentage: percentage,
              message: progress.stage || `Analyzing sentiment data... ${processedRecords}/${totalRecords}`
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
            title: ' Analysis Complete',
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
  );
}