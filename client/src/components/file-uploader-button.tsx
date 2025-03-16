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
  const currentCount = useRef<number>(0);

  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

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

          console.log('Starting upload with', lines, 'records');

          // Reset current count
          currentCount.current = 0;

          // Start upload process
          setIsUploading(true);

          // Initialize at 0
          updateUploadProgress({
            totalRecords: lines,
            processedRecords: 0,
            message: 'Starting analysis...',
            status: 'uploading',
            percentage: 0
          });

          // Ensure initial state is visible
          await sleep(500);

          const result = await uploadCSV(file, async (progress) => {
            // Extract record number from progress
            let nextRecord = 0;
            const recordMatch = progress.stage?.match(/record (\d+) of (\d+)/i);

            if (recordMatch) {
              nextRecord = parseInt(recordMatch[1]);
            } else if (progress.processed) {
              nextRecord = Number(progress.processed);
            }

            // Ensure we're moving forward
            if (nextRecord > currentCount.current) {
              // Increment one by one with delays
              for (let i = currentCount.current + 1; i <= nextRecord; i++) {
                await sleep(200); // Force delay between updates

                currentCount.current = i;
                updateUploadProgress({
                  processedRecords: i,
                  totalRecords: lines,
                  percentage: Math.floor((i / lines) * 100),
                  message: `Processing record ${i} of ${lines}`,
                  status: 'uploading'
                });
              }
            }

            if (progress.error) {
              throw new Error(progress.error);
            }
          });

          if (result?.file && result?.posts) {
            // Show completion state
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

            // Refresh queries
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

    // Reset states after error
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