import { Upload, Loader2, AlertCircle, Ban } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDisasterContext } from '@/context/disaster-context';
// Rename imported context isUploading hook to avoid name conflicts
import { isUploading as checkIsUploading } from '@/lib/upload-persistence';
import { useToast } from '@/hooks/use-toast';
import { uploadCSV, checkForActiveSessions } from '@/lib/api';
import { 
  setUploadSessionId, 
  saveUploadProgress,
  startTrackingUpload
} from '@/lib/upload-persistence';
import { queryClient } from '@/lib/queryClient';
import { useEffect, useState, useRef } from 'react';
import { 
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger
} from "@/components/ui/tooltip";

interface FileUploaderButtonProps {
  onSuccess?: (data: any) => void;
  className?: string;
}

export function FileUploaderButton({ onSuccess, className }: FileUploaderButtonProps) {
  const { toast } = useToast();
  const { isUploading, setIsUploading, setUploadProgress } = useDisasterContext();
  const [isCheckingForUploads, setIsCheckingForUploads] = useState(true);
  const [uploadBlocked, setUploadBlocked] = useState(false);
  const [uploadBlockReason, setUploadBlockReason] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Periodically check for active uploads (every 20 seconds)
  useEffect(() => {
    const checkActive = async () => {
      try {
        setIsCheckingForUploads(true);
        const activeSessionId = await checkForActiveSessions();
        
        // Handle special error string from checkForActiveSessions
        if (activeSessionId === 'error') {
          setUploadBlocked(true);
          setUploadBlockReason('Unable to verify upload status. Please refresh the page and try again.');
          console.warn('Upload blocked: Error checking active sessions');
          return;
        }
        
        // Check if an upload was in progress before refresh using the persistence module
        const wasUploading = checkIsUploading();
        
        // If there's an active session that isn't showing in our UI,
        // block the upload and explain why
        if (activeSessionId && !isUploading) {
          setUploadBlocked(true);
          setUploadBlockReason('An upload is already in progress in another tab or window. Please wait for it to complete.');
          
          // Reconnect to the active upload session
          if (!isUploading) {
            setIsUploading(true);
          }
          
          console.warn('Upload blocked: Active session detected in database but not in local state');
        } else if (wasUploading && !isUploading) {
          // We had an upload in progress but it's not showing in the UI
          setUploadBlocked(true);
          setUploadBlockReason('An upload was in progress. Please refresh the page if you want to start a new one.');
          console.warn('Upload blocked: Active session detected in localStorage but not in local state');
        } else if (!activeSessionId && !wasUploading && uploadBlocked) {
          // Clear the block if there's no active session anywhere
          setUploadBlocked(false);
          setUploadBlockReason('');
        }
        
        // Simplified log to reduce console spam
        if (!uploadBlocked && !isUploading) {
          console.log('Active upload session check complete:', 
            activeSessionId ? `Session ${activeSessionId} active` : 'No active sessions');
        }
      } catch (error) {
        console.error('Error checking for active uploads:', error);
        // On error, take a conservative approach and block uploads
        setUploadBlocked(true);
        setUploadBlockReason('Unable to verify if an upload is in progress. Please try again in a moment.');
      } finally {
        setIsCheckingForUploads(false);
      }
    };
    
    // Initial check
    checkActive();
    
    // Set up periodic check
    const intervalId = setInterval(checkActive, 20000);
    
    // Clean up interval on unmount
    return () => clearInterval(intervalId);
  }, [isUploading, uploadBlocked]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];

    // Double-check if we already have an active upload
    // This prevents race conditions where multiple uploads could start nearly simultaneously
    const activeSessionId = await checkForActiveSessions();
    
    // Handle specific error case
    if (activeSessionId === 'error') {
      toast({
        title: 'Upload Not Allowed',
        description: 'Unable to verify upload status. Please refresh the page and try again.',
        variant: 'destructive',
      });
      event.target.value = '';
      return;
    }
    
    // Handle active upload case
    if (activeSessionId || isUploading || uploadBlocked) {
      toast({
        title: 'Upload Not Allowed',
        description: 'Another upload is already in progress. Please wait for it to complete.',
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
      // Disable further uploads immediately
      setUploadBlocked(true);
      
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
          lastBatchDuration: 0,
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
            lastBatchDuration: progress.processingStats?.lastBatchDuration || 0,
            averageSpeed: progress.processingStats?.averageSpeed || 0
          }
        };

        // Save progress in localStorage using our persistence module
        saveUploadProgress(currentProgress);
        
        // Update UI with the current progress
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
            lastBatchDuration: 0,
            averageSpeed: 0
          }
        });
        
        // Unblock uploads after a short delay
        setTimeout(() => {
          setUploadBlocked(false);
          setUploadBlockReason('');
        }, 1000);
      }, 2000);
    }
  };

  // Stop any click events when blocked or uploading
  const handleButtonClick = (e: React.MouseEvent) => {
    if (isUploading || isCheckingForUploads || uploadBlocked) {
      e.preventDefault();
      
      if (uploadBlocked && uploadBlockReason) {
        toast({
          title: 'Upload Blocked',
          description: uploadBlockReason,
          variant: 'destructive',
        });
      } else if (isUploading) {
        toast({
          title: 'Upload in Progress',
          description: 'Please wait for the current upload to complete.',
          variant: 'destructive',
        });
      }
    }
  };

  const isButtonDisabled = isUploading || isCheckingForUploads || uploadBlocked;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <motion.label
            whileHover={{ scale: isButtonDisabled ? 1 : 1.03 }}
            whileTap={{ scale: isButtonDisabled ? 1 : 0.97 }}
            className={`
              relative inline-flex items-center justify-center px-5 py-2.5 h-10
              ${isButtonDisabled
                ? 'bg-gray-500 cursor-not-allowed opacity-75' 
                : 'bg-gradient-to-r from-teal-500 to-emerald-500 hover:from-teal-600 hover:to-emerald-600 cursor-pointer'
              }
              text-white text-sm font-medium rounded-full
              transition-all duration-300
              shadow-md hover:shadow-lg
              overflow-hidden
              ${className}
            `}
            onClick={handleButtonClick}
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
              ) : uploadBlocked ? (
                <>
                  <Ban className="h-4 w-4 mr-2" />
                  <span>Upload Blocked</span>
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4 mr-2" />
                  <span>Upload Dataset</span>
                </>
              )}
            </div>

            {/* Only allow file selection when not uploading and not blocked */}
            {!isButtonDisabled && (
              <input 
                ref={fileInputRef}
                type="file" 
                className="hidden" 
                accept=".csv" 
                onChange={handleFileUpload}
              />
            )}
          </motion.label>
        </TooltipTrigger>
        <TooltipContent>
          {isCheckingForUploads 
            ? "Checking if another upload is in progress..." 
            : isUploading 
              ? "An upload is currently in progress. Please wait for it to complete." 
              : uploadBlocked
                ? uploadBlockReason
                : "Upload a CSV file for sentiment analysis"
          }
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}