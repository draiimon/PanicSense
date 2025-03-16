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
      message: 'Preparing file for analysis...',
      percentage: 0,
      processedRecords: 0,
      totalRecords: 0
    });

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

          // Set initial progress with line count
          updateUploadProgress({ 
            totalRecords: lines,
            processedRecords: 0,
            message: 'Starting analysis...' 
          });

          // Wait a moment for UI to update
          await new Promise(resolve => setTimeout(resolve, 100));

          const result = await uploadCSV(file, (progress) => {
            // IMPORTANT FIX: Use Number to ensure proper conversion rather than parseInt
            // This avoids issues with parseInt not handling decimal strings properly
            let processedRecords = Number(progress.processed) || 0;
            let totalRecords = Number(progress.total) || lines || 100;
            
            // Debug logs for root cause analysis
            console.log('DIRECT RAW VALUES:', {
              processedFromEvent: progress.processed,
              totalFromEvent: progress.total,
              convertedProcessed: processedRecords,
              convertedTotal: totalRecords
            });
            
            // Fix specifically for common case: when the backend says we're done,
            // ensure the progress shows the final state
            if (progress.stage?.includes('complete') || processedRecords >= totalRecords) {
              processedRecords = totalRecords;
            }
            
            // Calculate percentage, max 100%
            const percentage = Math.min(Math.round((processedRecords / totalRecords) * 100), 100);
            
            // Force the UI update by applying this on the next tick
            setTimeout(() => {
              // Call updateUploadProgress directly here for immediate effect
              updateUploadProgress({
                processedRecords,
                totalRecords,
                percentage,
                message: progress.stage || 'Processing...'
              });
            }, 0);
            
            // Log for debugging - IMPORTANT to know what the UI will display
            console.log('CRITICAL UI PROGRESS DATA:', { 
              processedRecords, 
              totalRecords, 
              percentage, 
              stage: progress.stage
            });

            // Update UI with the processed values
            updateUploadProgress({
              processedRecords, // This has been properly parsed as a number
              totalRecords,    // This has been properly parsed as a number
              percentage,
              message: progress.stage || `Processing records... ${processedRecords}/${totalRecords}`,
              status: progress.error ? 'error' : 'uploading'
            });

            if (progress.error) {
              throw new Error(progress.error);
            }
          });

          if (!result || !result.file || !result.posts) {
            throw new Error('Invalid response from server');
          }

          // Get the final count values
          const finalTotalRecords = lines || uploadProgress.totalRecords;
          const finalProcessedRecords = finalTotalRecords; // When complete, processed = total
          
          updateUploadProgress({
            status: 'success',
            message: 'Analysis Complete!',
            percentage: 100,
            // Make sure the counts show the final state (all records processed)
            processedRecords: finalProcessedRecords,
            totalRecords: finalTotalRecords
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

          // Reset upload state after a much longer delay to ensure progress is visible
          progressTimeout.current = setTimeout(() => {
            resetUploadProgress();
            // Only reset the isUploading state after the progress UI has been visible for a while
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
      // Reset file input
      event.target.value = '';
      
      // We'll let the success or error handlers manage the isUploading state
      // Don't reset it here as it could interrupt the loading UI
    }
  };

  const handleError = (error: unknown) => {
    console.error('File upload error:', error);

    updateUploadProgress({
      status: 'error',
      message: error instanceof Error ? error.message : 'Upload failed',
      percentage: 0
    });

    toast({
      title: 'Analysis Failed',
      description: error instanceof Error ? error.message : 'An unexpected error occurred during analysis',
      variant: 'destructive',
      duration: 7000,
    });

    progressTimeout.current = setTimeout(() => {
      resetUploadProgress();
      // Keep the error visible for longer
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