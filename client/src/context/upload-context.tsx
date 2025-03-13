
import { createContext, useContext, useState, ReactNode } from 'react';
import axios from 'axios';
import { useToast } from '@/components/ui/use-toast';

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

interface UploadContextType {
  status: UploadStatus;
  progress: string;
  uploadFile: (file: File) => Promise<any>;
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

  const uploadFile = async (file: File): Promise<any> => {
    if (status === 'uploading') return;
    
    try {
      setStatus('uploading');
      setProgress('Preparing upload...');
      
      // Start progress simulation
      simulateProgressUpdates();
      
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post('/api/upload-csv', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            updateProgress(`Uploading: ${percentCompleted}%`);
          }
        },
      });
      
      setStatus('success');
      setProgress('Upload complete');
      
      toast({
        title: "Success",
        description: `Analyzed ${response.data.posts.length} posts from ${file.name}`,
      });
      
      return response.data;
    } catch (error) {
      console.error('Upload error:', error);
      setStatus('error');
      setProgress('Upload failed');
      
      toast({
        title: "Error",
        description: "Failed to upload and process file. Please try again.",
        variant: "destructive",
      });
      
      return null;
    }
  };
  
  const reset = () => {
    setStatus('idle');
    setProgress('Preparing...');
  };

  return (
    <UploadContext.Provider value={{ status, progress, uploadFile, reset }}>
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
