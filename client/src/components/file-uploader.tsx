import { useState, useEffect, useRef } from 'react';
import { uploadCSV } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
import { queryClient } from '@/lib/queryClient';
import { useDisasterContext } from '@/context/disaster-context';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Loader2, CheckCircle, AlertCircle } from 'lucide-react';

interface FileUploaderProps {
  onSuccess?: (data: any) => void;
  className?: string;
}

export function FileUploader({ onSuccess, className }: FileUploaderProps) {
  const [uploadProgress, setUploadProgress] = useState('Preparing...');
  const [progressPercentage, setProgressPercentage] = useState(0);
  const [processedRecords, setProcessedRecords] = useState(0);
  const [totalRecords, setTotalRecords] = useState(0);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const { toast } = useToast();
  const { isUploading, setIsUploading } = useDisasterContext();
  const progressTimeout = useRef<NodeJS.Timeout>();

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (progressTimeout.current) {
        clearTimeout(progressTimeout.current);
      }
    };
  }, []);

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
    setUploadStatus('uploading');
    setProgressPercentage(0);
    setProcessedRecords(0);
    setTotalRecords(0);

    try {
      setUploadProgress('Reading file...');

      const reader = new FileReader();
      reader.onload = async (e) => {
        const text = e.target?.result as string;
        const lines = text.split('\n').length - 1;
        setTotalRecords(lines);

        try {
          const result = await uploadCSV(file, (progress) => {
            setProcessedRecords(progress.processed);
            setProgressPercentage(calculateProgress(progress.processed, lines));
            setUploadProgress(`Processing records (${progress.processed}/${lines})`);

            if (progress.error) {
              throw new Error(progress.error);
            }
          });

          setProgressPercentage(100);
          setUploadProgress('Analysis Complete');
          setUploadStatus('success');

          toast({
            title: 'Success!',
            description: `Analyzed ${result.posts.length} posts successfully`,
          });

          // Invalidate queries to refresh data
          queryClient.invalidateQueries({ queryKey: ['/api/sentiment-posts'] });
          queryClient.invalidateQueries({ queryKey: ['/api/analyzed-files'] });
          queryClient.invalidateQueries({ queryKey: ['/api/disaster-events'] });

          if (onSuccess) {
            onSuccess(result);
          }

          // Delay resetting the upload state
          progressTimeout.current = setTimeout(() => {
            setIsUploading(false);
            setUploadStatus('idle');
          }, 2000);

        } catch (error) {
          setUploadStatus('error');
          throw error;
        }
      };

      reader.readAsText(file);
    } catch (error) {
      setUploadStatus('error');
      toast({
        title: 'Upload failed',
        description: error instanceof Error ? error.message : 'An unexpected error occurred',
        variant: 'destructive',
      });

      // Reset state after error
      progressTimeout.current = setTimeout(() => {
        setIsUploading(false);
        setUploadStatus('idle');
      }, 2000);
    } finally {
      event.target.value = '';
    }
  };

  return (
    <div className={className}>
      <AnimatePresence mode="wait">
        {uploadStatus === 'idle' ? (
          <motion.label
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="inline-flex items-center justify-center w-[140px] h-[40px] bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white text-sm font-medium rounded-md cursor-pointer transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
          >
            <Upload className="h-5 w-5 mr-2" />
            Upload CSV
            <input 
              type="file" 
              className="hidden" 
              accept=".csv" 
              onChange={handleFileUpload}
              disabled={isUploading}
            />
          </motion.label>
        ) : (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex flex-col items-center p-4 bg-white border rounded-lg shadow-lg min-w-[300px]"
          >
            <div className="flex items-center mb-3">
              {uploadStatus === 'uploading' && (
                <Loader2 className="animate-spin h-5 w-5 mr-3 text-blue-600" />
              )}
              {uploadStatus === 'success' && (
                <CheckCircle className="h-5 w-5 mr-3 text-green-600" />
              )}
              {uploadStatus === 'error' && (
                <AlertCircle className="h-5 w-5 mr-3 text-red-600" />
              )}
              <span className={`font-medium ${
                uploadStatus === 'error' ? 'text-red-800' :
                uploadStatus === 'success' ? 'text-green-800' :
                'text-blue-800'
              }`}>
                {uploadProgress}
              </span>
            </div>

            <motion.div 
              className="w-full bg-gray-200 rounded-full h-2.5 mb-2 overflow-hidden"
            >
              <motion.div 
                className={`h-2.5 rounded-full ${
                  uploadStatus === 'error' ? 'bg-red-600' :
                  uploadStatus === 'success' ? 'bg-green-600' :
                  'bg-blue-600'
                }`}
                initial={{ width: 0 }}
                animate={{ width: `${progressPercentage}%` }}
                transition={{ duration: 0.3 }}
              />
            </motion.div>

            {totalRecords > 0 && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-xs text-gray-500 text-center"
              >
                Processed {processedRecords} of {totalRecords} records
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}