import { X } from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { cancelUpload } from "@/lib/api";

export function UploadProgressModal() {
  const { isUploading, setIsUploading, uploadProgress, setUploadProgress } = useDisasterContext();
  const [showDetailedProgress, setShowDetailedProgress] = useState(false);

  // Don't allow modal to be closed by clicking outside or pressing Escape
  // Instead handle the close button click
  const handleClose = async () => {
    // If upload is in progress, confirm cancellation
    if (isUploading && !uploadProgress.stage?.toLowerCase().includes("complete")) {
      const confirmCancel = window.confirm("Cancel the current upload?");
      
      if (confirmCancel) {
        try {
          // Cancel the upload on the server
          const result = await cancelUpload();
          
          if (result.success) {
            setUploadProgress({
              ...uploadProgress,
              stage: "Cancelled by user"
            });
            
            // Close modal after a short delay
            setTimeout(() => {
              setIsUploading(false);
            }, 1000);
          } else {
            alert(`Failed to cancel: ${result.message}`);
          }
        } catch (error) {
          console.error("Error cancelling upload:", error);
          alert("Failed to cancel the upload. Please try again.");
        }
      }
    } else {
      // If upload is complete, just close the modal
      setIsUploading(false);
    }
  };

  // If upload is complete, automatically close after a delay
  useEffect(() => {
    let closingTimer: NodeJS.Timeout;
    
    if (uploadProgress.stage?.toLowerCase().includes("complete") ||
        uploadProgress.stage?.toLowerCase().includes("error") ||
        uploadProgress.stage?.toLowerCase().includes("finished")) {
      closingTimer = setTimeout(() => {
        setIsUploading(false);
      }, 2000);
    }
    
    return () => {
      if (closingTimer) clearTimeout(closingTimer);
    };
  }, [uploadProgress.stage, setIsUploading]);

  // Calculate progress percentage
  const progressPercentage = uploadProgress.total > 0 
    ? Math.round((uploadProgress.processed / uploadProgress.total) * 100)
    : 0;
  
  // Calculate estimated time remaining
  const getTimeRemaining = () => {
    if (!uploadProgress.timeRemaining) return "Calculating...";
    
    const seconds = uploadProgress.timeRemaining;
    if (seconds <= 0) return "Almost done...";
    
    if (seconds < 60) return `${Math.round(seconds)} seconds`;
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds} minutes`;
  };

  return (
    <Dialog open={isUploading} onOpenChange={(open) => {
      if (!open) handleClose();
    }}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Processing CSV File</DialogTitle>
          <DialogDescription>
            We're analyzing the sentiment of your data. This might take a few moments.
          </DialogDescription>
        </DialogHeader>
        
        <div className="py-4">
          <div className="mb-4">
            <div className="flex justify-between mb-2">
              <p className="text-sm font-medium">{uploadProgress.stage || 'Preparing...'}</p>
              <p className="text-sm text-gray-500">{progressPercentage}%</p>
            </div>
            <Progress value={progressPercentage} />
          </div>
          
          <div className="grid grid-cols-2 gap-2 text-sm text-gray-500 mb-3">
            <div>Processed: {uploadProgress.processed} of {uploadProgress.total}</div>
            {uploadProgress.currentSpeed !== undefined && (
              <div className="text-right">Speed: {uploadProgress.currentSpeed.toFixed(1)} records/sec</div>
            )}
          </div>
          
          {/* Toggle for detailed stats */}
          <button 
            onClick={() => setShowDetailedProgress(!showDetailedProgress)}
            className="text-xs text-blue-600 hover:underline mb-2"
          >
            {showDetailedProgress ? 'Hide details' : 'Show details'}
          </button>
          
          {/* Detailed progress stats */}
          {showDetailedProgress && (
            <div className="bg-gray-50 p-3 rounded-md text-xs space-y-1 mb-4">
              <div className="grid grid-cols-2">
                <span>Success:</span>
                <span className="text-green-600 font-medium">
                  {uploadProgress.processingStats?.successCount || 0} records
                </span>
              </div>
              <div className="grid grid-cols-2">
                <span>Errors:</span>
                <span className="text-red-600 font-medium">
                  {uploadProgress.processingStats?.errorCount || 0} records
                </span>
              </div>
              <div className="grid grid-cols-2">
                <span>Avg. Speed:</span>
                <span className="font-medium">
                  {uploadProgress.processingStats?.averageSpeed?.toFixed(1) || 0} rec/sec
                </span>
              </div>
              {uploadProgress.timeRemaining !== undefined && (
                <div className="grid grid-cols-2">
                  <span>Est. Time:</span>
                  <span className="font-medium">{getTimeRemaining()}</span>
                </div>
              )}
            </div>
          )}
          
          <div className="flex justify-end">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleClose}
              className="relative"
            >
              {uploadProgress.stage?.toLowerCase().includes("complete") ||
               uploadProgress.stage?.toLowerCase().includes("finished") ?
                'Close' : 'Cancel'}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}