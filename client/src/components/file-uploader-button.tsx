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

    // More strict file validation
    if (!file.name.toLowerCase().endsWith('.csv')) {
      toast({
        title: 'Invalid File Format',
        description: 'Please upload a CSV file containing disaster-related data.',
        variant: 'destructive',
      });
      return;
    }

    // Check file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
      toast({
        title: 'File Too Large',
        description: 'Please upload a CSV file smaller than 50MB.',
        variant: 'destructive',
      });
      return;
    }

    setIsUploading(true);
    updateUploadProgress({
      status: 'uploading',
      message: 'Counting total records...',
      percentage: 0,
      processedRecords: 0,
      totalRecords: 0
    });

    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const text = e.target?.result as string;
          const lines = text.split('\n').filter(line => line.trim()).length;

          // Validate minimum content
          if (lines < 2) {
            throw new Error('CSV file appears to be empty or malformed. Please check the file content.');
          }

          updateUploadProgress({ 
            totalRecords: lines - 1, // Subtract header row
            message: 'Starting sentiment analysis...',
            percentage: 5
          });

          const result = await uploadCSV(file, (progress) => {
            // Handle progress updates
            if (progress.processed !== undefined) {
              const processedRecords = progress.processed;
              const totalRecords = lines - 1; // Subtract header row
              const percentage = Math.min(Math.round((processedRecords / totalRecords) * 90) + 5, 95);

              updateUploadProgress({
                processedRecords,
                totalRecords,
                percentage,
                message: progress.stage || `Analyzing records: ${processedRecords}/${totalRecords} (${percentage}%)`,
                status: progress.error ? 'error' : 'uploading'
              });

              if (progress.error) {
                throw new Error(progress.error);
              }
            }
          });

          if (!result || !result.file || !result.posts) {
            throw new Error('Invalid response from server');
          }

          updateUploadProgress({
            status: 'success',
            message: `Successfully analyzed ${result.posts.length} records!`,
            percentage: 100,
            processedRecords: result.posts.length,
            totalRecords: lines - 1
          });

          toast({
            title: 'Analysis Complete',
            description: `Successfully analyzed ${result.posts.length} records with sentiment data`,
            duration: 5000,
          });

          // Invalidate queries to refresh data
          queryClient.invalidateQueries({ queryKey: ['/api/sentiment-posts'] });
          queryClient.invalidateQueries({ queryKey: ['/api/analyzed-files'] });
          queryClient.invalidateQueries({ queryKey: ['/api/disaster-events'] });

          if (onSuccess) {
            onSuccess(result);
          }

          // Reset upload state after delay
          progressTimeout.current = setTimeout(() => {
            resetUploadProgress();
          }, 2000);

        } catch (error) {
          handleError(error);
        }
      };

      reader.onerror = () => {
        handleError(new Error('Failed to read file'));
      };

      reader.readAsText(file);
    } catch (error) {
      handleError(error);
    } finally {
      // Reset file input
      event.target.value = '';

      // Ensure uploading state is eventually reset
      setTimeout(() => {
        setIsUploading(false);
      }, 3000);
    }
  };

  const handleError = (error: unknown) => {
    console.error('File upload error:', error);

    updateUploadProgress({
      status: 'error',
      message: error instanceof Error ? error.message : 'Upload failed',
      percentage: 0,
      processedRecords: 0,
      totalRecords: 0
    });

    toast({
      title: 'Analysis Failed',
      description: error instanceof Error ? error.message : 'An unexpected error occurred during analysis',
      variant: 'destructive',
      duration: 7000,
    });

    progressTimeout.current = setTimeout(() => {
      resetUploadProgress();
      setIsUploading(false);
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
        ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
        ${className}
      `}
    >
      <Upload className="h-5 w-5 mr-2" />
      {isUploading ? (
        <span>
          Analyzing... {uploadProgress.processedRecords}/{uploadProgress.totalRecords}
        </span>
      ) : (
        'Upload Dataset'
      )}
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