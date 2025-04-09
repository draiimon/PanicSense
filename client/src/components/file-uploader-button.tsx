import { Upload, Loader2, RefreshCw } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDisasterContext } from '@/context/disaster-context';
import { useToast } from '@/hooks/use-toast';
import { uploadCSV, checkForActiveSessions, cleanupErrorSessions, resetUploadSessions } from '@/lib/api';
import { queryClient } from '@/lib/queryClient';
import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';

interface FileUploaderButtonProps {
  onSuccess?: (data: any) => void;
  className?: string;
}

export function FileUploaderButton({ onSuccess, className }: FileUploaderButtonProps) {
  const { toast } = useToast();
  const { isUploading, setIsUploading, setUploadProgress } = useDisasterContext();
  const [isCheckingForUploads, setIsCheckingForUploads] = useState(true);
  const [isCleaningUpSessions, setIsCleaningUpSessions] = useState(false);
  const [isResettingUploads, setIsResettingUploads] = useState(false);
  const [sessionCleanedCount, setSessionCleanedCount] = useState(0);

  // Automatically cleanup error sessions on mount to help with flickering issues
  useEffect(() => {
    const cleanupOnMount = async () => {
      try {
        console.log("🧹 Auto-cleaning error sessions on component mount");
        const result = await cleanupErrorSessions();
        if (result.success) {
          setSessionCleanedCount(result.clearedCount);
          if (result.clearedCount > 0) {
            console.log(`✅ Auto-cleaned ${result.clearedCount} error/stale sessions on mount`);
          }
        }
      } catch (error) {
        console.error("Error during auto-cleanup:", error);
      }
    };
    
    cleanupOnMount();
  }, []);

  // Check for active uploads on mount (after cleanup)
  useEffect(() => {
    const checkActive = async () => {
      try {
        setIsCheckingForUploads(true);
        const activeSessionId = await checkForActiveSessions();
        
        // If there's an active session and we're not already in upload mode,
        // the DisasterContext will handle setting up the connection
        
        if (activeSessionId) {
          // Ensure the upload modal is displayed if we have an active session
          setIsUploading(true);
          console.log('Active upload session detected:', `Session ${activeSessionId} active`);
        } else {
          console.log('Active upload session check complete: No active sessions');
        }
      } catch (error) {
        console.error('Error checking for active uploads:', error);
      } finally {
        setIsCheckingForUploads(false);
      }
    };
    
    checkActive();
  }, [setIsUploading]);
  
  // Handler for manual cleanup of error sessions
  const handleCleanupErrorSessions = async () => {
    try {
      setIsCleaningUpSessions(true);
      const result = await cleanupErrorSessions();
      if (result.success) {
        setSessionCleanedCount(result.clearedCount);
        toast({
          title: 'Sessions Cleaned',
          description: `Successfully cleaned ${result.clearedCount} error or stale sessions`,
          duration: 3000,
        });
        
        // Refresh data
        queryClient.invalidateQueries({ queryKey: ['/api/sentiment-posts'] });
        queryClient.invalidateQueries({ queryKey: ['/api/analyzed-files'] });
        queryClient.invalidateQueries({ queryKey: ['/api/disaster-events'] });
        
        // Force reload of any active session info
        await checkForActiveSessions();
      }
    } catch (error) {
      console.error('Error cleaning sessions:', error);
      toast({
        title: 'Cleanup Failed',
        description: error instanceof Error ? error.message : 'Failed to clean up sessions',
        variant: 'destructive',
      });
    } finally {
      setIsCleaningUpSessions(false);
    }
  };
  
  // Handler for emergency reset of upload system
  const handleResetUploads = async () => {
    try {
      setIsResettingUploads(true);
      const result = await resetUploadSessions();
      if (result.success) {
        toast({
          title: 'Upload System Reset',
          description: 'Successfully reset all upload sessions',
          duration: 3000,
        });
        
        // Refresh all data
        queryClient.invalidateQueries();
        
        // Clear any uploading state
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
      }
    } catch (error) {
      console.error('Error resetting uploads:', error);
      toast({
        title: 'Reset Failed',
        description: error instanceof Error ? error.message : 'Failed to reset upload system',
        variant: 'destructive',
      });
    } finally {
      setIsResettingUploads(false);
    }
  };

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
      // Set uploading state and progress in a single update
      setUploadProgress({ 
        processed: 0, 
        total: 100, 
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
      
      // Set uploading flag without delay
      setIsUploading(true);

      // Set localStorage flag for persistence across refreshes
      localStorage.setItem('isUploading', 'true');
      localStorage.setItem('uploadStartTime', Date.now().toString());
      
      const result = await uploadCSV(file, (progress) => {
        // Enhanced progress tracking with timestamp
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
          },
          timestamp: Date.now(), // Add timestamp for ordered updates
          savedAt: Date.now()    // Add timestamp for freshness check
        };

        console.log('Progress update:', currentProgress);
        
        // Update the UI
        setUploadProgress(currentProgress);
        
        // Store in localStorage for persistence across refreshes
        localStorage.setItem('uploadProgress', JSON.stringify(currentProgress));
        localStorage.setItem('lastProgressTimestamp', Date.now().toString());
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
      
      // ENHANCED ERROR HANDLING - Clean up ALL localStorage state
      localStorage.removeItem('isUploading');
      localStorage.removeItem('uploadProgress');
      localStorage.removeItem('uploadSessionId');
      localStorage.removeItem('uploadStartTime');
      localStorage.removeItem('lastProgressTimestamp');
      localStorage.removeItem('lastDatabaseCheck');
      localStorage.removeItem('serverRestartProtection');
      localStorage.removeItem('serverRestartTimestamp');
      localStorage.removeItem('cooldownActive');
      localStorage.removeItem('cooldownStartedAt');
      localStorage.removeItem('lastTabSync');
      
      // Force server cleanup on error
      fetch('/api/reset-upload-sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      }).catch(() => { /* ignore */ });
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
        
        // Properly clear ALL localStorage items related to upload
        localStorage.removeItem('isUploading');
        localStorage.removeItem('uploadProgress');
        localStorage.removeItem('uploadSessionId');
        localStorage.removeItem('uploadStartTime');
        localStorage.removeItem('lastProgressTimestamp');
        localStorage.removeItem('lastDatabaseCheck');
        localStorage.removeItem('serverRestartProtection');
        localStorage.removeItem('serverRestartTimestamp');
        localStorage.removeItem('cooldownActive');
        localStorage.removeItem('cooldownStartedAt');
        localStorage.removeItem('lastTabSync');
        
        console.log('Upload completed, cleared all localStorage items');
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