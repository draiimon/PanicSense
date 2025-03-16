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
    setIsUploading,
    updateUploadProgress,
    resetUploadProgress
  } = useDisasterContext();
  const progressTimeout = useRef<NodeJS.Timeout>();
  const lastProgress = useRef<number>(0);

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

    try {
      // Read file content to get line count
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const text = e.target?.result as string;
          const lines = text.split('\n').length - 1;

          console.log('Starting file upload with', lines, 'records');

          // Reset progress tracking
          lastProgress.current = 0;

          // Start upload process
          setIsUploading(true);

          // Initialize progress at 0
          updateUploadProgress({
            totalRecords: lines,
            processedRecords: 0,
            message: 'Starting analysis...',
            status: 'uploading',
            percentage: 0
          });

          // Wait briefly to ensure initial state is shown
          await new Promise(resolve => setTimeout(resolve, 100));

          const result = await uploadCSV(file, (progress) => {
            // Log raw progress data for debugging
            console.log('Raw progress update:', {
              stage: progress.stage,
              processed: progress.processed,
              total: progress.total,
              currentProgress: lastProgress.current
            });

            // Extract current record number
            let currentRecord = 0;

            // Try to get record number from stage message
            const recordMatch = progress.stage?.match(/record (\d+) of (\d+)/i);
            if (recordMatch) {
              currentRecord = parseInt(recordMatch[1]);
            } else if (progress.processed) {
              currentRecord = Number(progress.processed);
            }

            // If there's a big jump, animate through intermediate values
            if (currentRecord - lastProgress.current > 1) {
              console.log('Detected progress jump:', {
                from: lastProgress.current,
                to: currentRecord
              });

              // Update in smaller increments
              const step = Math.max(1, Math.floor((currentRecord - lastProgress.current) / 5));
              let animatedRecord = lastProgress.current;

              const animateProgress = () => {
                animatedRecord = Math.min(currentRecord, animatedRecord + step);

                updateUploadProgress({
                  processedRecords: animatedRecord,
                  totalRecords: lines,
                  percentage: Math.floor((animatedRecord / lines) * 100),
                  message: `Processing record ${animatedRecord} of ${lines}`,
                  status: 'uploading'
                });

                if (animatedRecord < currentRecord) {
                  setTimeout(animateProgress, 100);
                }
              };

              animateProgress();
            } else {
              // Normal single increment update
              updateUploadProgress({
                processedRecords: currentRecord,
                totalRecords: lines,
                percentage: Math.floor((currentRecord / lines) * 100),
                message: progress.stage || `Processing record ${currentRecord} of ${lines}`,
                status: 'uploading'
              });
            }

            // Store current progress for next update
            lastProgress.current = currentRecord;

            if (progress.error) {
              throw new Error(progress.error);
            }
          });

          // Handle successful upload
          if (result?.file && result?.posts) {
            updateUploadProgress({
              status: 'success',
              message: 'Analysis Complete!',
              percentage: 100,
              processedRecords: lines,
              totalRecords: lines
            });

            toast({
              title: 'Upload Complete',
              description: `Successfully analyzed ${result.posts.length} posts`,
              duration: 5000,
            });

            // Refresh data
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
              }, 2000);
            }, 3000);
          }
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
    console.error('Upload error:', error);

    updateUploadProgress({
      status: 'error',
      message: error instanceof Error ? error.message : 'Upload failed',
      percentage: 0,
      processedRecords: 0,
      totalRecords: 0
    });

    toast({
      title: 'Upload Failed',
      description: error instanceof Error ? error.message : 'An unexpected error occurred',
      variant: 'destructive',
      duration: 5000,
    });

    // Keep error state visible briefly before resetting
    progressTimeout.current = setTimeout(() => {
      resetUploadProgress();
      setTimeout(() => {
        setIsUploading(false);
      }, 2000);
    }, 3000);
  };

  return (
    <motion.label
      className={`
        inline-flex items-center justify-center px-6 py-3
        bg-gradient-to-r from-blue-600 to-indigo-600
        hover:from-blue-700 hover:to-indigo-700
        text-white text-sm font-medium rounded-full
        cursor-pointer transition-all duration-300
        shadow-lg hover:shadow-xl transform hover:-translate-y-0.5
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
      />
    </motion.label>
  );
}