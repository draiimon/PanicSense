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
      const baseDelay = 3000; // Base delay of 3 seconds between records
      const randomDelay = () => baseDelay + (Math.random() * 1000); // Add 0-1 second random variation
      const batchSize = 6; // Process in batches of 6 like the actual process

      let currentItem = 0;
      let currentBatch = 1;

      const processNextItem = () => {
        if (currentItem >= totalItems) {
          setIsComplete(true);
          onComplete?.();
          return;
        }

        currentItem++;
        setProcessedItems(currentItem);
        setProgress((currentItem / totalItems) * 100);

        // Simulate batch processing stages
        if (currentItem % batchSize === 0) {
          setStage(`Completed batch ${currentBatch} - pausing before next batch`);
          currentBatch++;
          // Add longer delay between batches (5 seconds)
          setTimeout(() => {
            if (currentItem < totalItems) {
              setStage(`Starting batch ${currentBatch} - processing records ${currentItem + 1} to ${Math.min(currentItem + batchSize, totalItems)}`);
              setTimeout(processNextItem, 1000); // Start next batch after 1 second
            }
          }, 5000);
        } else {
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
        }
      };

      // Start processing after initial delay
      setStage("Initializing analysis for 20 records...");
      setTimeout(processNextItem, 2000);
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