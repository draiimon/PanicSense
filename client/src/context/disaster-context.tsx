import { createContext, ReactNode, useContext, useState } from "react";
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
  batchNumber: undefined,
  totalBatches: undefined,
  batchProgress: undefined,
  currentSpeed: undefined,
  timeRemaining: undefined,
  processingStats: undefined
};

export function DisasterContextProvider({ children }: { children: ReactNode }) {
  // State
  const [selectedDisasterType, setSelectedDisasterType] = useState<string>("All");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>(initialProgress);

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