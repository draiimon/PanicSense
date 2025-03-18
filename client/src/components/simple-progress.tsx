import { useState, useEffect } from 'react';
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface SimpleProgressProps {
  totalItems?: number;
  isProcessing: boolean;
  onComplete?: () => void;
  currentProgress?: { processed: number; stage: string; total: number; } | null;
}

export function SimpleProgress({ totalItems = 20, isProcessing, onComplete, currentProgress }: SimpleProgressProps) {
  const [progress, setProgress] = useState(0);
  const [processedItems, setProcessedItems] = useState(0);
  const [avgSpeed, setAvgSpeed] = useState(0);
  const [startTime] = useState(Date.now());
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (!isProcessing) {
      setProgress(0);
      setProcessedItems(0);
      setAvgSpeed(0);
      setIsComplete(false);
      return;
    }

    // Update based on actual progress from Python process
    if (currentProgress) {
      const progressPercent = (currentProgress.processed / currentProgress.total) * 100;
      setProgress(progressPercent);
      setProcessedItems(currentProgress.processed);

      // Calculate real average speed
      const elapsedSeconds = (Date.now() - startTime) / 1000;
      const newSpeed = elapsedSeconds > 0 ? processedItems / elapsedSeconds : 0;
      setAvgSpeed(Number(newSpeed.toFixed(1)));

      if (progressPercent >= 100) {
        setIsComplete(true);
        onComplete?.();
      }
    }
  }, [isProcessing, currentProgress, totalItems, onComplete, startTime, processedItems]);

  return (
    <div className="w-full space-y-2">
      <Progress 
        value={progress} 
        className={cn(
          "transition-all duration-300",
          isComplete && "bg-green-100 [&>[role=progressbar]]:bg-green-500"
        )}
      />
      <div className="flex justify-between text-sm text-muted-foreground">
        <span>
          {currentProgress?.stage || `Processed: ${processedItems}/${totalItems}`}
        </span>
        <span>
          Average Speed: {avgSpeed} records/s
        </span>
      </div>
    </div>
  );
}