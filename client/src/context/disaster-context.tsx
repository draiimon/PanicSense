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

interface UploadProgress {
  status: 'idle' | 'uploading' | 'success' | 'error';
  message: string;
  percentage: number;
  processedRecords: number;
  totalRecords: number;
  error?: string;
}

const initialUploadProgress: UploadProgress = {
  status: 'idle',
  message: '',
  percentage: 0,
  processedRecords: 0,
  totalRecords: 0,
  error: undefined,
};

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

  // Upload states
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
  updateUploadProgress: (progress: Partial<UploadProgress>) => void;
  resetUploadProgress: () => void;

  // Refresh function
  refreshData: () => void;
}

const DisasterContext = createContext<DisasterContextType | undefined>(undefined);

export function DisasterContextProvider({ children }: { children: ReactNode }) {
  // State for filters and upload
  const [selectedDisasterType, setSelectedDisasterType] = useState<string>("All");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>(initialUploadProgress);

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
    if (!sentimentCounts[post.sentiment]) {
      sentimentCounts[post.sentiment] = 0;
    }
    sentimentCounts[post.sentiment]++;
  });

  const dominantSentiment = Object.entries(sentimentCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "Neutral";

  // Calculate average model confidence
  const totalConfidence = sentimentPosts.reduce((sum, post) => sum + post.confidence, 0);
  const modelConfidence = sentimentPosts.length > 0 ? totalConfidence / sentimentPosts.length : 0;

  // Update upload progress
  const updateUploadProgress = (progress: Partial<UploadProgress>) => {
    setUploadProgress(prev => ({
      ...prev,
      ...progress,
      // Ensure percentage is calculated correctly
      percentage: progress.totalRecords && progress.processedRecords !== undefined
        ? Math.round((progress.processedRecords / progress.totalRecords) * 100)
        : prev.percentage
    }));
  };

  // Reset function
  const resetUploadProgress = () => {
    setUploadProgress(initialUploadProgress);
    setIsUploading(false);
  };

  // Refresh function for fetching all data
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
        updateUploadProgress,
        resetUploadProgress,
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