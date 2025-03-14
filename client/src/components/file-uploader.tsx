import { useState, useEffect } from 'react';
import { uploadCSV } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
import { queryClient } from '@/lib/queryClient';
import { useDisasterContext } from '@/context/disaster-context';

interface FileUploaderProps {
  onSuccess?: (data: any) => void;
  className?: string;
}

export function FileUploader({ onSuccess, className }: FileUploaderProps) {
  const [uploadProgress, setUploadProgress] = useState('Preparing...');
  const [progressPercentage, setProgressPercentage] = useState(0);
  const [processedRecords, setProcessedRecords] = useState(0);
  const [totalRecords, setTotalRecords] = useState(0);
  const [loadingDots, setLoadingDots] = useState('');
  const { toast } = useToast();
  const { isUploading, setIsUploading } = useDisasterContext();

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
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isUploading]);

  const calculateProgress = (processed: number, total: number) => {
    if (total === 0) return 0;
    const progress = (processed / total) * 100;
    return Math.min(Math.round(progress), 100);
  };

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
    setProgressPercentage(0);
    setProcessedRecords(0);
    setTotalRecords(0);

    try {
      // Start file reading
      setUploadProgress('Reading file...');

      const reader = new FileReader();
      reader.onload = async (e) => {
        const text = e.target?.result as string;
        const lines = text.split('\n').length - 1; // -1 for header
        setTotalRecords(lines);

        try {
          const result = await uploadCSV(file, (progress) => {
            setProcessedRecords(progress.processed);
            setProgressPercentage(calculateProgress(progress.processed, lines));
            setUploadProgress(`Processing records (${progress.processed}/${lines})`);
          });

          // Set to 100% when complete
          setProgressPercentage(100);
          setUploadProgress('Analysis Complete');

          toast({
            title: 'File uploaded successfully',
            description: `Analyzed ${result.posts.length} posts`,
          });

          // Invalidate queries to refresh data
          queryClient.invalidateQueries({ queryKey: ['/api/sentiment-posts'] });
          queryClient.invalidateQueries({ queryKey: ['/api/analyzed-files'] });
          queryClient.invalidateQueries({ queryKey: ['/api/disaster-events'] });

          if (onSuccess) {
            onSuccess(result);
          }

          // Short delay before resetting the upload state
          setTimeout(() => {
            setIsUploading(false);
          }, 1000);
        } catch (error) {
          throw error;
        }
      };

      reader.readAsText(file);
    } catch (error) {
      toast({
        title: 'Upload failed',
        description: error instanceof Error ? error.message : 'An unexpected error occurred',
        variant: 'destructive',
      });
      setIsUploading(false);
    } finally {
      // Reset the input value to allow re-uploading the same file
      event.target.value = '';
    }
  };

  return (
    <div className={className}>
      {isUploading ? (
        <div className="flex flex-col items-center p-4 bg-blue-50 border border-blue-200 rounded-lg min-w-[300px]">
          <div className="flex items-center mb-3">
            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span className="text-blue-800 font-medium">
              {uploadProgress}{loadingDots} ({progressPercentage}%)
            </span>
          </div>

          <div className="w-full bg-gray-200 rounded-full h-2.5 mb-2">
            <div 
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-300 ease-in-out" 
              style={{ width: `${progressPercentage}%` }}
            />
          </div>

          {totalRecords > 0 && (
            <div className="text-xs text-gray-500 text-center">
              Processed {processedRecords} of {totalRecords} records
            </div>
          )}
        </div>
      ) : (
        <label className="inline-flex items-center justify-center w-[140px] h-[40px] bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-md cursor-pointer transition-colors">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          Upload CSV
          <input 
            type="file" 
            className="hidden" 
            accept=".csv" 
            onChange={handleFileUpload}
            disabled={isUploading}
          />
        </label>
      )}
    </div>
  );
}