import { createContext, ReactNode, useContext, useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
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
  modelConfidence: number;

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

  // WebSocket setup
  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const socket = new WebSocket(wsUrl);

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'progress') {
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
    queryFn: getSentimentPosts
  });

  const { 
    data: disasterEvents = [], 
    isLoading: isLoadingDisasterEvents,
    error: errorDisasterEvents,
    refetch: refetchDisasterEvents
  } = useQuery({ 
    queryKey: ['/api/disaster-events'],
    queryFn: getDisasterEvents
  });

  const { 
    data: analyzedFiles = [], 
    isLoading: isLoadingAnalyzedFiles,
    error: errorAnalyzedFiles,
    refetch: refetchAnalyzedFiles
  } = useQuery({ 
    queryKey: ['/api/analyzed-files'],
    queryFn: getAnalyzedFiles
  });

  // Calculate stats
  const activeDiastersCount = disasterEvents.length;
  const analyzedPostsCount = sentimentPosts.length;

  // Calculate dominant sentiment
  const sentimentCounts: Record<string, number> = {};
  sentimentPosts.forEach(post => {
    sentimentCounts[post.sentiment] = (sentimentCounts[post.sentiment] || 0) + 1;
  });

  const dominantSentiment = Object.entries(sentimentCounts)
    .sort((a, b) => b[1] - a[1])[0]?.[0] || "Neutral";

  // Calculate average model confidence
  const totalConfidence = sentimentPosts.reduce((sum, post) => sum + post.confidence, 0);
  const modelConfidence = sentimentPosts.length > 0 ? totalConfidence / sentimentPosts.length : 0;

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
        modelConfidence,
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