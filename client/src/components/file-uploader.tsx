import { useState, useEffect } from 'react';
import { uploadCSV } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
import { queryClient } from '@/lib/queryClient';

interface FileUploaderProps {
  onSuccess?: (data: any) => void;
  className?: string;
}

export function FileUploader({ onSuccess, className }: FileUploaderProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('Preparing...');
  const [loadingDots, setLoadingDots] = useState('');
  const { toast } = useToast();

  // Create animated loading dots effect
  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    if (isUploading) {
      intervalId = setInterval(() => {
        setLoadingDots(prev => {
          if (prev.length >= 3) return '';
          return prev + '.';
        });
      }, 500);
      
      // Simulate progression of loading phases
      const phases = [
        { message: 'Uploading file', delay: 1000 },
        { message: 'Processing data', delay: 2000 },
        { message: 'Analyzing with Groq API', delay: 3000 },
        { message: 'Detecting languages', delay: 4000 },
        { message: 'Running sentiment analysis', delay: 5000 }
      ];
      
      let timeout: NodeJS.Timeout;
      phases.forEach(({message, delay}) => {
        timeout = setTimeout(() => {
          if (isUploading) setUploadProgress(message);
        }, delay);
      });
      
      return () => {
        clearInterval(intervalId);
        clearTimeout(timeout);
      };
    }
  }, [isUploading]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    if (!file.name.toLowerCase().endsWith('.csv')) {
      toast({
        title: 'Invalid file type',
        description: 'Please upload a CSV file',
        variant: 'destructive',
      });
      return;
    }

    setIsUploading(true);
    try {
      const result = await uploadCSV(file);
      
      toast({
        title: 'File uploaded successfully',
        description: `Analyzed ${result.posts.length} posts`,
      });
      
      // Invalidate queries to refresh data
      queryClient.invalidateQueries({ queryKey: ['/api/sentiment-posts'] });
      queryClient.invalidateQueries({ queryKey: ['/api/analyzed-files'] });
      
      if (onSuccess) {
        onSuccess(result);
      }
    } catch (error) {
      toast({
        title: 'Upload failed',
        description: error instanceof Error ? error.message : 'An unexpected error occurred',
        variant: 'destructive',
      });
    } finally {
      setIsUploading(false);
      // Reset the input value to allow re-uploading the same file
      event.target.value = '';
    }
  };

  return (
    <div className={className}>
      <label className={`inline-flex items-center px-4 py-2 ${isUploading ? 'bg-blue-400' : 'bg-blue-600 hover:bg-blue-700'} text-white text-sm font-medium rounded-md cursor-pointer transition-colors`}>
        {isUploading ? (
          <>
            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Processing...
          </>
        ) : (
          <>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            Upload CSV
          </>
        )}
        <input 
          type="file" 
          className="hidden" 
          accept=".csv" 
          onChange={handleFileUpload}
          disabled={isUploading}
        />
      </label>
    </div>
  );
}
