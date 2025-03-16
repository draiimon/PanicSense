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

    // Validate file format
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

    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const text = e.target?.result as string;
          const lines = text.split('\n').length - 1;

          // Validate minimum content
          if (lines < 2) {
            throw new Error('CSV file appears to be empty or malformed. Please check the file content.');
          }

          // Initialize progress with 0 records processed
          updateUploadProgress({ 
            totalRecords: lines,
            processedRecords: 0,
            message: 'Starting analysis...',
            status: 'uploading',
            percentage: 0
          });

          // Small delay to ensure initial state is shown
          await new Promise(resolve => setTimeout(resolve, 100));

          const result = await uploadCSV(file, (progress) => {
            let currentRecord = 0;

            // Extract current record number from stage message
            const recordMatch = progress.stage?.match(/record (\d+) of (\d+)/i);

            if (recordMatch) {
              currentRecord = parseInt(recordMatch[1]);
            } else if (progress.stage?.includes('complete')) {
              currentRecord = lines;
            } else {
              // If no explicit record number, use processed percentage
              const percentMatch = progress.stage?.match(/(\d+)%/);
              if (percentMatch) {
                const percent = parseInt(percentMatch[1]);
                currentRecord = Math.floor((percent / 100) * lines);
              }
            }

            // Ensure values are within valid range
            currentRecord = Math.max(0, Math.min(currentRecord, lines));

            // Calculate percentage
            const percentage = Math.round((currentRecord / lines) * 100);

            // Update progress state
            updateUploadProgress({
              processedRecords: currentRecord,
              totalRecords: lines,
              percentage,
              message: progress.stage || `Processing record ${currentRecord} of ${lines}`,
              status: progress.error ? 'error' : 'uploading'
            });

            if (progress.error) {
              throw new Error(progress.error);
            }
          });

          if (!result || !result.file || !result.posts) {
            throw new Error('Invalid response from server');
          }

          // Show completion state
          updateUploadProgress({
            status: 'success',
            message: 'Analysis Complete!',
            percentage: 100,
            processedRecords: lines,
            totalRecords: lines
          });

          toast({
            title: 'Analysis Complete',
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

          // Reset states after delays
          progressTimeout.current = setTimeout(() => {
            resetUploadProgress();
            setTimeout(() => {
              setIsUploading(false);
            }, 3000);
          }, 7000);

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
      event.target.value = '';
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
      setTimeout(() => {
        setIsUploading(false);
      }, 5000);
    }, 5000);
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
      {isUploading ? 'Analyzing...' : 'Upload Dataset'}
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