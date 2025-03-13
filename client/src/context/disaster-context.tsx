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

interface DisasterContextType {
  // Data
  sentimentPosts: SentimentPost[];
  disasterEvents: DisasterEvent[];
  analyzedFiles: AnalyzedFile[];
  
  // Loading states
  isLoadingSentimentPosts: boolean;
  isLoadingDisasterEvents: boolean;
  isLoadingAnalyzedFiles: boolean;
  
  // Error states
  errorSentimentPosts: Error | null;
  errorDisasterEvents: Error | null;
  errorAnalyzedFiles: Error | null;
  
  // Stats
  activeDiastersCount: number;
  analyzedPostsCount: number;
  dominantSentiment: string;
  aiConfidence: number;
  
  // Filters
  selectedDisasterType: string;
  setSelectedDisasterType: (type: string) => void;
  
  // Refresh function
  refreshData: () => void;
}

const DisasterContext = createContext<DisasterContextType | undefined>(undefined);

export function DisasterContextProvider({ children }: { children: ReactNode }) {
  // State for filters
  const [selectedDisasterType, setSelectedDisasterType] = useState<string>("All");
  
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
  
  // Calculate average AI confidence
  const totalConfidence = sentimentPosts.reduce((sum, post) => sum + post.confidence, 0);
  const aiConfidence = sentimentPosts.length > 0 ? totalConfidence / sentimentPosts.length : 0;

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
        errorSentimentPosts: errorSentimentPosts as Error | null,
        errorDisasterEvents: errorDisasterEvents as Error | null,
        errorAnalyzedFiles: errorAnalyzedFiles as Error | null,
        activeDiastersCount,
        analyzedPostsCount,
        dominantSentiment,
        aiConfidence,
        selectedDisasterType,
        setSelectedDisasterType,
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
