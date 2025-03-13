import React, { createContext, useContext, useState, ReactNode } from 'react';

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