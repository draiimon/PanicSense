import { Upload, Loader2, BrainCircuit, FileText, Info } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDisasterContext } from '@/context/disaster-context';
import { useToast } from '@/hooks/use-toast';
import { uploadCSVHybrid, checkForActiveSessions, cleanupErrorSessions } from '@/lib/api';
import { queryClient } from '@/lib/queryClient';
import { useEffect, useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Checkbox } from '@/components/ui/checkbox';
import axios from 'axios';

interface HybridFileUploaderButtonProps {
  onSuccess?: (data: any) => void;
  className?: string;
  id?: string;
}

interface ModelInfo {
  name: string;
  path: string;
  size: number;
  modified: string;
  sizeFormatted: string;
  type: string;
  location: string;
}

export function HybridFileUploaderButton({ onSuccess, className, id }: HybridFileUploaderButtonProps) {
  const { toast } = useToast();
  const { isUploading, setIsUploading, setUploadProgress } = useDisasterContext();
  const [isCheckingForUploads, setIsCheckingForUploads] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [textColumn, setTextColumn] = useState('text');
  const [validateData, setValidateData] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Check for active uploads on mount
  useEffect(() => {
    const checkActive = async () => {
      try {
        setIsCheckingForUploads(true);
        const activeSessionId = await checkForActiveSessions();
        
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

  // Fetch available models
  const fetchModels = async () => {
    try {
      setIsLoadingModels(true);
      const response = await axios.get('/api/hybrid-model/models');
      
      if (response.data && response.data.models) {
        setAvailableModels(response.data.models);
        
        // If models exist, select the first one by default
        if (response.data.models.length > 0) {
          setSelectedModel(response.data.models[0].name);
        }
      } else {
        console.log('No models available');
        setAvailableModels([]);
      }
    } catch (error) {
      console.error('Error fetching models:', error);
      toast({
        title: 'Error',
        description: 'Failed to fetch available models',
        variant: 'destructive',
      });
      setAvailableModels([]);
    } finally {
      setIsLoadingModels(false);
    }
  };
  
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
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
    
    // Store the selected file
    setSelectedFile(file);
    
    // Open the dialog and fetch available models
    setDialogOpen(true);
    fetchModels();
  };
  
  const handleStartAnalysis = async () => {
    if (!selectedFile) {
      toast({
        title: 'No File Selected',
        description: 'Please select a CSV file for analysis.',
        variant: 'destructive',
      });
      return;
    }
    
    try {
      // Close the dialog
      setDialogOpen(false);
      
      // Set uploading state and progress in a single update
      setUploadProgress({ 
        processed: 0, 
        total: 100, 
        stage: 'Initializing hybrid model...',
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
      localStorage.setItem('uploadModelType', 'hybrid');
      
      // Prepare upload options
      const uploadOptions = {
        modelName: selectedModel || undefined,
        textColumn: textColumn || 'text',
        validate: validateData
      };
      
      console.log('Starting hybrid model analysis with options:', uploadOptions);
      
      const result = await uploadCSVHybrid(selectedFile, (progress) => {
        // Enhanced progress tracking with timestamp
        const currentProgress = {
          processed: Number(progress.processed) || 0,
          total: Number(progress.total) || 0,
          stage: progress.stage || 'Processing with Hybrid Neural Network...',
          batchNumber: progress.batchNumber || 0,
          totalBatches: progress.totalBatches || 0,
          batchProgress: progress.batchProgress || 0,
          currentSpeed: progress.currentSpeed || 0,
          timeRemaining: progress.timeRemaining || 0,
          error: progress.error, // Preserve any error message from server
          autoCloseDelay: progress.autoCloseDelay, // Preserve autoCloseDelay from server
          processingStats: {
            successCount: progress.processingStats?.successCount || 0,
            errorCount: progress.processingStats?.errorCount || 0,
            averageSpeed: progress.processingStats?.averageSpeed || 0
          },
          timestamp: Date.now(), // Add timestamp for ordered updates
          savedAt: Date.now()    // Add timestamp for freshness check
        };

        console.log('Hybrid model progress update:', currentProgress);
        
        // Update the UI
        setUploadProgress(currentProgress);
        
        // Store in localStorage for persistence across refreshes
        localStorage.setItem('uploadProgress', JSON.stringify(currentProgress));
        localStorage.setItem('lastProgressTimestamp', Date.now().toString());
      });

      // Only show success toast and refresh data if we have real results
      if (result?.file && result?.posts) {
        // Only show toast if there are actual posts and not an error-recovery scenario
        if (result.posts.length > 0 && !result.errorRecovered) {
          // Show toast for successful completion
          toast({
            title: 'Hybrid Model Analysis Complete',
            description: `Successfully analyzed ${result.posts.length} posts with neural network`,
            duration: 5000,
          });
        } else {
          console.log('Skipping upload completion toast - no posts or error recovery mode');
        }

        // Refresh data quietly in background
        queryClient.invalidateQueries({ queryKey: ['/api/sentiment-posts'] });
        queryClient.invalidateQueries({ queryKey: ['/api/analyzed-files'] });
        queryClient.invalidateQueries({ queryKey: ['/api/disaster-events'] });

        if (onSuccess) {
          onSuccess(result);
        }
      }
    } catch (error) {
      console.error('Hybrid model upload error:', error);
      toast({
        title: 'Hybrid Model Analysis Failed',
        description: error instanceof Error ? error.message : 'Failed to analyze file with neural network',
        variant: 'destructive',
      });
      
      try {
        // Update the progress with error state to ensure it shows properly
        const errorProgress = {
          processed: 0,
          total: 10,
          stage: 'Upload Error',
          error: error instanceof Error ? error.message : 'Failed to upload file',
          timestamp: Date.now(),
          savedAt: Date.now(),
          autoCloseDelay: 0, // INSTANT CLOSE on error (let the component handle the minimal delay)
          processingStats: {
            successCount: 0,
            errorCount: 0,
            averageSpeed: 0
          }
        };
        
        // Update the UI with error state that will auto-close
        setUploadProgress(errorProgress);
        
        // Also save the error state to localStorage (will be auto-cleaned)
        localStorage.setItem('uploadProgress', JSON.stringify(errorProgress));
        
        // Force cleanup after 1 second to handle any UI race conditions
        setTimeout(() => {
          // Clean up localStorage
          localStorage.removeItem('uploadSessionId');
          localStorage.removeItem('uploadProgress');
          localStorage.removeItem('isUploading');
          localStorage.removeItem('uploadStartTime');
          localStorage.removeItem('uploadModelType');
          
          // Also call the API to ensure server is cleaned up
          cleanupErrorSessions().catch(e => console.error('Error in post-error cleanup:', e));
          
          console.log('🧹 POST-ERROR FORCED CLEANUP COMPLETE');
        }, 1000);
      } catch (cleanupError) {
        console.error('Error during error cleanup:', cleanupError);
      }
    } finally {
      event.target.value = '';
      
      // Don't automatically close the upload modal here
      // The UploadProgressModal component will handle auto-closing based on the autoCloseDelay parameter
      console.log('Hybrid model upload operation completed, the modal will auto-close based on server instructions');
    }
  };

  return (
    <motion.label
      id={id}
      whileHover={{ scale: isUploading || isCheckingForUploads ? 1 : 1.03 }}
      whileTap={{ scale: isUploading || isCheckingForUploads ? 1 : 0.97 }}
      className={`
        relative inline-flex items-center justify-center px-5 py-2.5 h-10
        ${isUploading 
          ? 'bg-gray-500 cursor-not-allowed opacity-70' 
          : 'bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-600 hover:to-indigo-700 cursor-pointer'
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
            <BrainCircuit className="h-4 w-4 mr-2" />
            <span>Neural Network Analysis</span>
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