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
  const [stage, setStage] = useState("");

  useEffect(() => {
    if (!isProcessing) {
      setProgress(0);
      setProcessedItems(0);
      setAvgSpeed(0);
      setIsComplete(false);
      setStage("");
      return;
    }

    // Simulate a slow, natural-feeling progress with random delays
    const simulateProgress = () => {
      const baseDelay = 2000; // Base delay of 2 seconds between records
      const randomDelay = () => baseDelay + (Math.random() * 1000); // Add 0-1 second random variation

      let currentItem = 0;

      const processNextItem = () => {
        if (currentItem >= totalItems) {
          setIsComplete(true);
          onComplete?.();
          return;
        }

        currentItem++;
        setProcessedItems(currentItem);
        setProgress((currentItem / totalItems) * 100);
        setStage(`Processing record ${currentItem}/${totalItems}`);

        // Calculate realistic-looking average speed
        const elapsedSeconds = (Date.now() - startTime) / 1000;
        const speed = elapsedSeconds > 0 ? currentItem / elapsedSeconds : 0;
        // Add some random variation to speed
        const randomizedSpeed = speed * (0.8 + Math.random() * 0.4); // ±20% variation
        setAvgSpeed(Number(randomizedSpeed.toFixed(1)));

        // Schedule next item with random delay
        if (currentItem < totalItems) {
          setTimeout(processNextItem, randomDelay());
        }
      };

      // Start processing after initial delay
      setTimeout(processNextItem, 500);
    };

    simulateProgress();
  }, [isProcessing, totalItems, onComplete, startTime]);

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
          {stage || `Processed: ${processedItems}/${totalItems}`}
        </span>
        <span>
          Average Speed: {avgSpeed} records/s
        </span>
      </div>
    </div>
  );
}