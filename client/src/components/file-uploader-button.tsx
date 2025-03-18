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
  const { setIsUploading, setUploadProgress } = useDisasterContext();

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];

    if (!file.name.toLowerCase().endsWith('.csv')) {
      toast({
        title: 'Invalid File Format',
        description: 'Please upload a CSV file.',
        variant: 'destructive',
      });
      return;
    }

    try {
      // Reset sequence
      setIsUploading(false);
      await new Promise(resolve => setTimeout(resolve, 100));
      setUploadProgress({ 
        processed: 0, 
        total: 0, 
        stage: 'Initializing...',
        currentSpeed: 0,
        timeRemaining: 0,
        batchNumber: 0,
        totalBatches: 0,
        batchProgress: 0,
        processingStats: {
          successCount: 0,
          errorCount: 0,
          averageSpeed: 0
        }
      });
      await new Promise(resolve => setTimeout(resolve, 100));
      setIsUploading(true);

      const result = await uploadCSV(file, (progress) => {
        // Enhanced progress tracking
        const currentProgress = {
          processed: Number(progress.processed) || 0,
          total: Number(progress.total) || 0,
          stage: progress.stage || 'Processing...',
          batchNumber: progress.batchNumber || 0,
          totalBatches: progress.totalBatches || 0,
          batchProgress: progress.batchProgress || 0,
          currentSpeed: progress.currentSpeed || 0,
          timeRemaining: progress.timeRemaining || 0,
          processingStats: {
            successCount: progress.processingStats?.successCount || 0,
            errorCount: progress.processingStats?.errorCount || 0,
            averageSpeed: progress.processingStats?.averageSpeed || 0
          }
        };

        console.log('Progress update:', currentProgress);
        setUploadProgress(currentProgress);
      });

      if (result?.file && result?.posts) {
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
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast({
        title: 'Upload Failed',
        description: error instanceof Error ? error.message : 'Failed to upload file',
        variant: 'destructive',
      });
    } finally {
      event.target.value = '';

      // Show completion for a moment before closing
      setTimeout(() => {
        setIsUploading(false);
        setUploadProgress({ 
          processed: 0, 
          total: 0, 
          stage: '',
          currentSpeed: 0,
          timeRemaining: 0,
          batchNumber: 0,
          totalBatches: 0,
          batchProgress: 0,
          processingStats: {
            successCount: 0,
            errorCount: 0,
            averageSpeed: 0
          }
        });
      }, 2000);
    }
  };

  return (
    <motion.label
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={`
        relative inline-flex items-center justify-center px-6 py-3
        bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-600
        hover:from-blue-700 hover:via-indigo-700 hover:to-blue-700
        text-white text-sm font-medium rounded-full
        cursor-pointer transition-all duration-300
        shadow-lg hover:shadow-xl
        overflow-hidden
        ${className}
      `}
    >
      {/* Animated background */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-400/20 via-indigo-400/20 to-blue-400/20 animate-gradient-x" />

      {/* Content */}
      <div className="relative flex items-center">
        <Upload className="h-5 w-5 mr-2" />
        Upload Dataset
      </div>

      <input 
        type="file" 
        className="hidden" 
        accept=".csv" 
        onChange={handleFileUpload}
      />

      <style>
        {`
          @keyframes gradient-x {
            0% {
              background-position: 0% 50%;
            }
            50% {
              background-position: 100% 50%;
            }
            100% {
              background-position: 0% 50%;
            }
          }
          .animate-gradient-x {
            animation: gradient-x 3s ease infinite;
            background-size: 200% 100%;
          }
        `}
      </style>
    </motion.label>
  );
}