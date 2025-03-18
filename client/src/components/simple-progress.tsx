import { useState, useEffect } from 'react';
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface SimpleProgressProps {
  totalItems: number; // Required prop, no default value
  isProcessing: boolean;
  onComplete?: () => void;
  stage?: string; // The current stage message from the system
}

export function SimpleProgress({ totalItems, isProcessing, onComplete, stage }: SimpleProgressProps) {
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

    // Extract numbers from stage message if available
    if (stage) {
      const matches = stage.match(/(\d+)\/(\d+)/);
      if (matches) {
        const current = parseInt(matches[1]);
        setProcessedItems(current);
        setProgress((current / totalItems) * 100);

        // Calculate actual speed
        const elapsedSeconds = (Date.now() - startTime) / 1000;
        const speed = elapsedSeconds > 0 ? current / elapsedSeconds : 0;
        setAvgSpeed(Number(speed.toFixed(1)));

        if (current >= totalItems) {
          setIsComplete(true);
          onComplete?.();
        }
      }
    }
  }, [isProcessing, stage, totalItems, onComplete, startTime]);

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
          {stage || `Completed record ${processedItems}/${totalItems}`}
        </span>
        <span>
          Average Speed: {avgSpeed} records/s
        </span>
      </div>
    </div>
  );
}