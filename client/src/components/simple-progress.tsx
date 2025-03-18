import { useState, useEffect } from 'react';
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface SimpleProgressProps {
  totalItems?: number;
  isProcessing: boolean;
  onComplete?: () => void;
}

export function SimpleProgress({ totalItems = 20, isProcessing, onComplete }: SimpleProgressProps) {
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

    // Simulate realistic progress with variable speeds
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsComplete(true);
          onComplete?.();
          return 100;
        }

        // Simulate varying speeds
        const randomIncrement = Math.random() * 2 + 1; // 1-3% increment
        const newProgress = Math.min(prev + randomIncrement, 100);

        // Update processed items based on progress
        const newProcessedItems = Math.floor((newProgress / 100) * totalItems);
        setProcessedItems(newProcessedItems);

        // Calculate average speed
        const elapsedSeconds = (Date.now() - startTime) / 1000;
        const newSpeed = elapsedSeconds > 0 ? newProcessedItems / elapsedSeconds : 0;
        setAvgSpeed(Number(newSpeed.toFixed(1)));

        return newProgress;
      });
    }, 200); // Update every 200ms for smooth animation

    return () => clearInterval(interval);
  }, [isProcessing, totalItems, onComplete]);

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
          Processed: {processedItems}/{totalItems}
        </span>
        <span>
          Average Speed: {avgSpeed} records/s
        </span>
      </div>
    </div>
  );
}