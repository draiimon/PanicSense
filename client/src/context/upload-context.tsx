import React, { createContext, useContext, useState } from 'react';

interface UploadContextType {
  isUploading: boolean;
  progress: number;
  startUpload: () => void;
  finishUpload: () => void;
  setProgress: (progress: number) => void;
}

const UploadContext = createContext<UploadContextType | undefined>(undefined);

export const UploadProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);

  const startUpload = () => {
    setIsUploading(true);
    setProgress(0);
  };

  const finishUpload = () => {
    setIsUploading(false);
    setProgress(0);
  };

  return (
    <UploadContext.Provider value={{ isUploading, progress, startUpload, finishUpload, setProgress }}>
      {children}
    </UploadContext.Provider>
  );
};

export const useUpload = (): UploadContextType => {
  const context = useContext(UploadContext);
  if (context === undefined) {
    throw new Error('useUpload must be used within an UploadProvider');
  }
  return context;
}