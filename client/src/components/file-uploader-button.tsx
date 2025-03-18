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
      // Clear any previous upload state
      setIsUploading(false);
      
      // Wait a moment before starting the new upload to ensure reset of any tracking state
      await new Promise(resolve => setTimeout(resolve, 50));
      
      // Start the new upload process with a clear initialization state
      setIsUploading(true);
      // Set initializing stage which will trigger our reset in the upload progress modal
      setUploadProgress({ processed: 0, total: 0, stage: 'Initializing new upload...' });

      const result = await uploadCSV(file, (progress) => {
        setUploadProgress({
          processed: Number(progress.processed) || 0,
          total: Number(progress.total) || 0,
          stage: progress.stage || 'Processing...',
          batchNumber: progress.batchNumber,
          totalBatches: progress.totalBatches,
          batchProgress: progress.batchProgress,
          currentSpeed: progress.currentSpeed,
          timeRemaining: progress.timeRemaining,
          processingStats: progress.processingStats
        });
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

      // Keep progress visible briefly after completion
      setTimeout(() => {
        setIsUploading(false);
        setUploadProgress({ processed: 0, total: 0, stage: '' });
      }, 2000);
    }
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