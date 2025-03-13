
import React, { createContext, useContext, useState, ReactNode } from 'react';
import { uploadCSV } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
import { queryClient } from '@/lib/queryClient';

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

interface UploadContextType {
  status: UploadStatus;
  progress: string;
  uploadFile: (file: File) => Promise<void>;
  reset: () => void;
}

const UploadContext = createContext<UploadContextType | undefined>(undefined);

export function UploadProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<UploadStatus>('idle');
  const [progress, setProgress] = useState<string>('Preparing...');
  const { toast } = useToast();

  const updateProgress = (message: string) => {
    setProgress(message);
  };

  const simulateProgressUpdates = () => {
    const phases = [
      { message: 'Uploading file', delay: 1000 },
      { message: 'Processing data', delay: 2000 },
      { message: 'Analyzing text', delay: 3000 },
      { message: 'Detecting languages', delay: 4000 },
      { message: 'Running sentiment analysis', delay: 5000 }
    ];
    
    phases.forEach(({message, delay}) => {
      setTimeout(() => {
        if (status === 'uploading') updateProgress(message);
      }, delay);
    });
  };

  const uploadFile = async (file: File) => {
    if (status === 'uploading') return;
    
    setStatus('uploading');
    setProgress('Preparing...');
    simulateProgressUpdates();
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await uploadCSV(formData);
      
      // Invalidate queries to refresh data
      queryClient.invalidateQueries({ queryKey: ['/api/analyzed-files'] });
      queryClient.invalidateQueries({ queryKey: ['/api/sentiment-posts'] });
      queryClient.invalidateQueries({ queryKey: ['/api/disaster-events'] });
      
      toast({
        title: "Upload Successful",
        description: `Analyzed ${response.posts.length} entries with sentiment data.`,
      });
      
      setStatus('success');
      
      return response;
    } catch (error) {
      console.error('Upload error:', error);
      toast({
        title: "Upload Failed",
        description: error instanceof Error ? error.message : "Failed to upload file",
        variant: "destructive",
      });
      
      setStatus('error');
    }
  };

  const reset = () => {
    setStatus('idle');
    setProgress('Preparing...');
  };

  return (
    <UploadContext.Provider value={{ 
      status, 
      progress, 
      uploadFile, 
      reset 
    }}>
      {children}
    </UploadContext.Provider>
  );
}

export function useUpload() {
  const context = useContext(UploadContext);
  if (context === undefined) {
    throw new Error('useUpload must be used within an UploadProvider');
  }
  return context;
}
