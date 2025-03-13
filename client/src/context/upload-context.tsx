
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
import React, { createContext, useContext, useState } from 'react';

interface UploadContextType {
  isUploading: boolean;
  setIsUploading: (uploading: boolean) => void;
  uploadProgress: number;
  setUploadProgress: (progress: number) => void;
  uploadError: string | null;
  setUploadError: (error: string | null) => void;
  uploadFile: (file: File) => Promise<any>;
}

const UploadContext = createContext<UploadContextType | undefined>(undefined);

export const useUpload = () => {
  const context = useContext(UploadContext);
  if (!context) {
    throw new Error('useUpload must be used within an UploadProvider');
  }
  return context;
};

export const UploadProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const uploadFile = async (file: File) => {
    try {
      setIsUploading(true);
      setUploadProgress(0);
      setUploadError(null);
      
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('/api/upload-csv', {
        method: 'POST',
        body: formData,
        // Simulate progress for better UX
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(progress);
          }
        },
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }
      
      setUploadProgress(100);
      return await response.json();
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Unknown error occurred');
      throw error;
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <UploadContext.Provider
      value={{
        isUploading,
        setIsUploading,
        uploadProgress,
        setUploadProgress,
        uploadError,
        setUploadError,
        uploadFile,
      }}
    >
      {children}
    </UploadContext.Provider>
  );
};
import React, { createContext, useContext, useState } from 'react';

interface UploadContextType {
  isUploading: boolean;
  progress: number;
  error: string | null;
  startUpload: () => void;
  finishUpload: (success: boolean, errorMsg?: string) => void;
  setProgress: (progress: number) => void;
}

const UploadContext = createContext<UploadContextType | undefined>(undefined);

export function useUpload() {
  const context = useContext(UploadContext);
  if (context === undefined) {
    throw new Error('useUpload must be used within an UploadProvider');
  }
  return context;
}

export function UploadProvider({ children }: { children: React.ReactNode }) {
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const startUpload = () => {
    setIsUploading(true);
    setProgress(0);
    setError(null);
  };

  const finishUpload = (success: boolean, errorMsg?: string) => {
    setIsUploading(false);
    setProgress(0);
    if (!success && errorMsg) {
      setError(errorMsg);
    } else {
      setError(null);
    }
  };

  return (
    <UploadContext.Provider
      value={{
        isUploading,
        progress,
        error,
        startUpload,
        finishUpload,
        setProgress,
      }}
    >
      {children}
    </UploadContext.Provider>
  );
}
