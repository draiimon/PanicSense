import { Upload, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDisasterContext } from '@/context/disaster-context';
import { useToast } from '@/hooks/use-toast';
import { uploadCSV } from '@/lib/api';
import { queryClient } from '@/lib/queryClient';
import { useRef } from 'react';
import { 
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger
} from "@/components/ui/tooltip";

interface FileUploaderButtonProps {
  onSuccess?: (data: any) => void;
  className?: string;
}

export function FileUploaderButton({ onSuccess, className }: FileUploaderButtonProps) {
  const { toast } = useToast();
  const { isUploading, setIsUploading, setUploadProgress } = useDisasterContext();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];

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
      // Set uploading state
      setIsUploading(true);
      
      // Initialize progress
      setUploadProgress({ 
        processed: 0, 
        total: 100, // Default total
        stage: 'Initializing...',
        processingStats: {
          successCount: 0,
          errorCount: 0,
          averageSpeed: 0
        }
      });

      const result = await uploadCSV(file, (progress) => {
        // Set progress updates
        setUploadProgress({
          processed: Number(progress.processed) || 0,
          total: Number(progress.total) || 100,
          stage: progress.stage || 'Processing...',
          processingStats: {
            successCount: progress.processingStats?.successCount || 0,
            errorCount: progress.processingStats?.errorCount || 0,
            averageSpeed: progress.processingStats?.averageSpeed || 0
          }
        });
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
      setUploadProgress({
        processed: 0,
        total: 100,
        stage: 'Upload failed. Please try again.',
        processingStats: {
          successCount: 0,
          errorCount: 1,
          averageSpeed: 0
        }
      });
      toast({
        title: 'Upload Failed',
        description: error instanceof Error ? error.message : 'Failed to upload file',
        variant: 'destructive',
      });
    } finally {
      event.target.value = '';
      
      // Show completion for a moment before resetting
      setTimeout(() => {
        setIsUploading(false);
      }, 2000);
    }
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <motion.label
            whileHover={{ scale: isUploading ? 1 : 1.03 }}
            whileTap={{ scale: isUploading ? 1 : 0.97 }}
            className={`
              relative inline-flex items-center justify-center px-5 py-2.5 h-10
              ${isUploading
                ? 'bg-gray-500 cursor-not-allowed opacity-75' 
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
              {isUploading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4 mr-2" />
                  <span>Upload Dataset</span>
                </>
              )}
            </div>

            {/* Only allow file selection when not uploading */}
            {!isUploading && (
              <input 
                ref={fileInputRef}
                type="file" 
                className="hidden" 
                accept=".csv" 
                onChange={handleFileUpload}
              />
            )}
          </motion.label>
        </TooltipTrigger>
        <TooltipContent>
          {isUploading 
            ? "Processing your file. Please wait..." 
            : "Upload a CSV file for sentiment analysis"}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}