import { createContext, ReactNode, useContext, useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { 
  getSentimentPosts, 
  getDisasterEvents, 
  getAnalyzedFiles,
  SentimentPost,
  DisasterEvent,
  AnalyzedFile
} from "@/lib/api";

// Define ProcessingStats interface
interface ProcessingStats {
  successCount: number;
  errorCount: number;
  averageSpeed: number;
}

// Define UploadProgress interface with all properties
interface UploadProgress {
  processed: number;
  total: number;
  stage: string;
  batchNumber?: number;
  totalBatches?: number;
  batchProgress?: number;
  currentSpeed?: number;
  timeRemaining?: number;
  processingStats?: ProcessingStats;
}

interface DisasterContextType {
  // Data
  sentimentPosts: SentimentPost[];
  disasterEvents: DisasterEvent[];
  analyzedFiles: AnalyzedFile[];

  // Loading states
  isLoadingSentimentPosts: boolean;
  isLoadingDisasterEvents: boolean;
  isLoadingAnalyzedFiles: boolean;
  isUploading: boolean;

  // Upload progress
  uploadProgress: UploadProgress;

  // Error states
  errorSentimentPosts: Error | null;
  errorDisasterEvents: Error | null;
  errorAnalyzedFiles: Error | null;

  // Stats
  activeDiastersCount: number;
  analyzedPostsCount: number;
  dominantSentiment: string;
  dominantDisaster: string;
  modelConfidence: number;
  
  // Sentiment statistics
  dominantSentimentPercentage?: number;
  secondDominantSentiment?: string | null;
  secondDominantSentimentPercentage?: number;
  sentimentPercentages?: Record<string, number>;
  
  // Disaster statistics
  dominantDisasterPercentage?: number;
  secondDominantDisaster?: string | null;
  secondDominantDisasterPercentage?: number;
  disasterPercentages?: Record<string, number>;

  // Filters
  selectedDisasterType: string;
  setSelectedDisasterType: (type: string) => void;

  // Upload state management
  setIsUploading: (state: boolean) => void;
  setUploadProgress: (progress: UploadProgress) => void;

  // Refresh function
  refreshData: () => void;
}

const DisasterContext = createContext<DisasterContextType | undefined>(undefined);

const initialProgress: UploadProgress = {
  processed: 0,
  total: 0,
  stage: '',
  batchNumber: 0,
  totalBatches: 0,
  batchProgress: 0,
  currentSpeed: 0,
  timeRemaining: 0,
  processingStats: {
    successCount: 0,
    errorCount: 0,
    averageSpeed: 0
  }
};

export function DisasterContextProvider({ children }: { children: ReactNode }) {
  // State
  const [selectedDisasterType, setSelectedDisasterType] = useState<string>("All");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>(initialProgress);

  // Get toast for notifications
  const { toast } = useToast();
  
  // WebSocket setup
  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const socket = new WebSocket(wsUrl);

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle sentiment feedback updates (for real-time UI updates)
        if (data.type === 'feedback-update') {
          console.log('WebSocket feedback update received:', data);
          
          // This will trigger a refresh of all data including sentiment posts
          refreshData();
          
          // Display a toast notification about the sentiment update
          toast({
            title: 'Model Updated',
            description: `Sentiment model has been trained with new feedback.`,
            variant: 'default',
          });
        }
        // Handle specific post update messages
        else if (data.type === 'post-updated') {
          console.log('WebSocket post update received:', data);
          
          // Dispatch a custom event for sentiment changes that the UI can listen for
          const updateEvent = new CustomEvent('sentiment-change', { 
            detail: data.data 
          });
          window.dispatchEvent(updateEvent);
          console.log('Sentiment change event dispatched:', updateEvent);
          
          // Force an immediate refresh to update the UI with the new sentiment
          refreshData();
          
          // Display a toast notification about the post update with more specific information
          let description = 'Post has been updated successfully.';
          if (data.data?.updates?.sentiment) {
            description = `Sentiment changed to ${data.data.updates.sentiment}`;
          } else if (data.data?.updates?.location) {
            description = `Location updated to ${data.data.updates.location}`;
          } else if (data.data?.updates?.disasterType) {
            description = `Disaster type updated to ${data.data.updates.disasterType}`;
          }
          
          toast({
            title: 'Post Updated',
            description,
            variant: 'default',
          });
        }
        // Handle progress updates for file processing
        else if (data.type === 'progress') {
          console.log('WebSocket progress update:', data);

          // Extract progress info from Python service
          const pythonProgress = data.progress;
          if (pythonProgress && typeof pythonProgress === 'object') {
            // Parse numbers from the progress message
            const matches = pythonProgress.stage?.match(/(\d+)\/(\d+)/);
            const currentRecord = matches ? parseInt(matches[1]) : 0;
            const totalRecords = matches ? parseInt(matches[2]) : pythonProgress.total || 0;

            // Calculate actual progress percentage
            const processedCount = pythonProgress.processed || currentRecord;

            setUploadProgress(prev => ({
              ...prev,
              processed: processedCount,
              total: totalRecords || prev.total,
              stage: pythonProgress.stage || prev.stage,
              batchNumber: currentRecord,
              totalBatches: totalRecords,
              batchProgress: totalRecords > 0 ? Math.round((processedCount / totalRecords) * 100) : 0,
              currentSpeed: pythonProgress.currentSpeed || prev.currentSpeed,
              timeRemaining: pythonProgress.timeRemaining || prev.timeRemaining,
              processingStats: {
                successCount: processedCount,
                errorCount: pythonProgress.processingStats?.errorCount || 0,
                averageSpeed: pythonProgress.processingStats?.averageSpeed || 0
              }
            }));
          }
        }
      } catch (error) {
        console.error('Error processing WebSocket message:', error);
      }
    };

    socket.onopen = () => {
      console.log('WebSocket connected');
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      socket.close();
    };
  }, []);

  // Queries
  const { 
    data: sentimentPosts = [], 
    isLoading: isLoadingSentimentPosts,
    error: errorSentimentPosts,
    refetch: refetchSentimentPosts
  } = useQuery({ 
    queryKey: ['/api/sentiment-posts'],
    queryFn: getSentimentPosts,
    staleTime: 10 * 60 * 1000, // 10 minutes
    retry: 3 // Retry 3 times if failed
  });

  const { 
    data: disasterEvents = [], 
    isLoading: isLoadingDisasterEvents,
    error: errorDisasterEvents,
    refetch: refetchDisasterEvents
  } = useQuery({ 
    queryKey: ['/api/disaster-events'],
    queryFn: getDisasterEvents,
    staleTime: 10 * 60 * 1000, // 10 minutes
    retry: 3 // Retry 3 times if failed
  });

  const { 
    data: analyzedFiles = [], 
    isLoading: isLoadingAnalyzedFiles,
    error: errorAnalyzedFiles,
    refetch: refetchAnalyzedFiles
  } = useQuery({ 
    queryKey: ['/api/analyzed-files'],
    queryFn: getAnalyzedFiles,
    staleTime: 10 * 60 * 1000, // 10 minutes
    retry: 3 // Retry 3 times if failed
  });

  // Calculate stats with safety checks for array data
  const activeDiastersCount = Array.isArray(disasterEvents) ? disasterEvents.length : 0;
  const analyzedPostsCount = Array.isArray(sentimentPosts) ? sentimentPosts.length : 0;

  // Calculate dominant sentiment with proper array check and percentages
  const sentimentCounts: Record<string, number> = {};
  const totalPosts = Array.isArray(sentimentPosts) ? sentimentPosts.length : 0;
  
  if (totalPosts > 0) {
    sentimentPosts.forEach(post => {
      sentimentCounts[post.sentiment] = (sentimentCounts[post.sentiment] || 0) + 1;
    });
  }

  // Sort sentiments by count from highest to lowest
  const sortedSentiments = Object.entries(sentimentCounts)
    .sort((a, b) => b[1] - a[1]);
    
  // Get the dominant sentiment
  const dominantSentiment = sortedSentiments.length > 0 
    ? sortedSentiments[0]?.[0] 
    : "Neutral";
    
  // Calculate percentages for each sentiment
  const sentimentPercentages = Object.fromEntries(
    Object.entries(sentimentCounts).map(([sentiment, count]) => [
      sentiment, 
      totalPosts > 0 ? Math.round((count / totalPosts) * 100) : 0
    ])
  );
  
  // Calculate dominant sentiment percentage
  const dominantSentimentPercentage = sentimentPercentages[dominantSentiment] || 0;
  
  // Calculate second most dominant sentiment if available
  const secondDominantSentiment = sortedSentiments.length > 1 
    ? sortedSentiments[1]?.[0] 
    : null;
  const secondDominantSentimentPercentage = secondDominantSentiment 
    ? sentimentPercentages[secondDominantSentiment] 
    : 0;
    
  // Calculate dominant disaster type with proper array check
  const disasterCounts: Record<string, number> = {};
  let validDisasterPostsCount = 0;
  
  if (totalPosts > 0) {
    sentimentPosts.forEach(post => {
      if (post.disasterType && 
          post.disasterType !== "Not Specified" && 
          post.disasterType !== "NONE" && 
          post.disasterType !== "None" && 
          post.disasterType !== "null" && 
          post.disasterType !== "undefined") {
        disasterCounts[post.disasterType] = (disasterCounts[post.disasterType] || 0) + 1;
        validDisasterPostsCount++;
      }
    });
  }
  
  // Sort disaster types by count from highest to lowest
  const sortedDisasters = Object.entries(disasterCounts)
    .sort((a, b) => b[1] - a[1]);
  
  // Get the dominant disaster
  const dominantDisaster = sortedDisasters.length > 0 
    ? sortedDisasters[0]?.[0] 
    : "Unknown";
    
  // Calculate percentages for each disaster type
  const disasterPercentages = Object.fromEntries(
    Object.entries(disasterCounts).map(([disasterType, count]) => [
      disasterType, 
      validDisasterPostsCount > 0 ? Math.round((count / validDisasterPostsCount) * 100) : 0
    ])
  );
  
  // Calculate dominant disaster percentage
  const dominantDisasterPercentage = disasterPercentages[dominantDisaster] || 0;
  
  // Calculate second most dominant disaster if available
  const secondDominantDisaster = sortedDisasters.length > 1 
    ? sortedDisasters[1]?.[0] 
    : null;
  const secondDominantDisasterPercentage = secondDominantDisaster 
    ? disasterPercentages[secondDominantDisaster] 
    : 0;

  // Calculate average model confidence with safety checks
  const totalConfidence = Array.isArray(sentimentPosts) 
    ? sentimentPosts.reduce((sum, post) => sum + (post.confidence || 0), 0)
    : 0;
  const modelConfidence = Array.isArray(sentimentPosts) && sentimentPosts.length > 0 
    ? totalConfidence / sentimentPosts.length 
    : 0;

  // Refresh function
  const refreshData = () => {
    refetchSentimentPosts();
    refetchDisasterEvents();
    refetchAnalyzedFiles();
  };

  return (
    <DisasterContext.Provider
      value={{
        sentimentPosts,
        disasterEvents,
        analyzedFiles,
        isLoadingSentimentPosts,
        isLoadingDisasterEvents,
        isLoadingAnalyzedFiles,
        isUploading,
        uploadProgress,
        errorSentimentPosts: errorSentimentPosts as Error | null,
        errorDisasterEvents: errorDisasterEvents as Error | null,
        errorAnalyzedFiles: errorAnalyzedFiles as Error | null,
        activeDiastersCount,
        analyzedPostsCount,
        dominantSentiment,
        dominantDisaster,
        modelConfidence,
        // Sentiment statistics
        dominantSentimentPercentage,
        secondDominantSentiment,
        secondDominantSentimentPercentage,
        sentimentPercentages,
        // Disaster statistics
        dominantDisasterPercentage,
        secondDominantDisaster,
        secondDominantDisasterPercentage,
        disasterPercentages,
        // Filters
        selectedDisasterType,
        setSelectedDisasterType,
        setIsUploading,
        setUploadProgress,
        refreshData
      }}
    >
      {children}
    </DisasterContext.Provider>
  );
}

export function useDisasterContext() {
  const context = useContext(DisasterContext);
  if (context === undefined) {
    throw new Error("useDisasterContext must be used within a DisasterContextProvider");
  }
  return context;
}