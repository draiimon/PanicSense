import { Upload, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDisasterContext } from '@/context/disaster-context';
import { useToast } from '@/hooks/use-toast';
import { uploadCSV, checkForActiveSessions } from '@/lib/api';
import { queryClient } from '@/lib/queryClient';
import { useEffect, useState } from 'react';

interface FileUploaderButtonProps {
  onSuccess?: (data: any) => void;
  className?: string;
}

export function FileUploaderButton({ onSuccess, className }: FileUploaderButtonProps) {
  const { toast } = useToast();
  const { isUploading, setIsUploading, setUploadProgress } = useDisasterContext();
  const [isCheckingForUploads, setIsCheckingForUploads] = useState(true);

  // Check for active uploads on mount
  useEffect(() => {
    const checkActive = async () => {
      try {
        setIsCheckingForUploads(true);
        const activeSessionId = await checkForActiveSessions();
        
        // If there's an active session and we're not already in upload mode,
        // the DisasterContext will handle setting up the connection
        
        console.log('Active upload session check complete:', 
          activeSessionId ? `Session ${activeSessionId} active` : 'No active sessions');
      } catch (error) {
        console.error('Error checking for active uploads:', error);
      } finally {
        setIsCheckingForUploads(false);
      }
    };
    
    checkActive();
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];

    // Check if we already have an active upload
    if (isUploading) {
      toast({
        title: 'Upload in Progress',
        description: 'Please wait for the current upload to complete.',
        variant: 'destructive',
      });
      event.target.value = '';
      return;
    }

    if (!file.name.toLowerCase().endsWith('.csv')) {
      toast({
        title: 'Invalid File Format',
        description: 'Please upload a CSV file.',
        variant: 'destructive',
      });
      event.target.value = '';
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
      whileHover={{ scale: isUploading || isCheckingForUploads ? 1 : 1.03 }}
      whileTap={{ scale: isUploading || isCheckingForUploads ? 1 : 0.97 }}
      className={`
        relative inline-flex items-center justify-center px-5 py-2.5 h-10
        ${isUploading 
          ? 'bg-gray-500 cursor-not-allowed opacity-70' 
          : 'bg-gradient-to-r from-teal-500 to-emerald-500 hover:from-teal-600 hover:to-emerald-600 cursor-pointer'
        }
        text-white text-sm font-medium rounded-full
        transition-all duration-300
        shadow-md hover:shadow-lg
        overflow-hidden
        ${className}
      `}
    >
      {/* Content */}
      <div className="relative flex items-center justify-center">
        {isCheckingForUploads ? (
          <>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            <span>Checking...</span>
          </>
        ) : isUploading ? (
          <>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            <span>Upload in Progress</span>
          </>
        ) : (
          <>
            <Upload className="h-4 w-4 mr-2" />
            <span>Upload Dataset</span>
          </>
        )}
      </div>

      {/* Only allow file selection when not uploading */}
      {!isUploading && !isCheckingForUploads && (
        <input 
          type="file" 
          className="hidden" 
          accept=".csv" 
          onChange={handleFileUpload}
        />
      )}
    </motion.label>
  );
}