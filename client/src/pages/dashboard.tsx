import { useState } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { FileUploader } from "@/components/file-uploader";
import { StatusCard } from "@/components/dashboard/status-card";
import { SentimentChart } from "@/components/dashboard/sentiment-chart";
import { AffectedAreas } from "@/components/dashboard/affected-areas";
import { RecentPostsTable } from "@/components/dashboard/recent-posts-table";

export default function Dashboard() {
  const { 
    sentimentPosts,
    activeDiastersCount,
    analyzedPostsCount,
    dominantSentiment,
    aiConfidence,
    isLoadingSentimentPosts
  } = useDisasterContext();

  // Calculate sentiment distribution
  const sentimentCounts: Record<string, number> = {
    'Panic': 0,
    'Fear/Anxiety': 0,
    'Disbelief': 0,
    'Resilience': 0,
    'Neutral': 0
  };

  sentimentPosts.forEach(post => {
    if (sentimentCounts.hasOwnProperty(post.sentiment)) {
      sentimentCounts[post.sentiment]++;
    }
  });

  const totalPosts = sentimentPosts.length || 1; // Avoid division by zero
  const sentimentPercentages = Object.entries(sentimentCounts).map(([label, count]) => ({
    label,
    percentage: Math.round((count / totalPosts) * 100)
  }));

  // Mock affected areas data (normally would be derived from sentimentPosts with location data)
  const affectedAreas = [
    { name: "Metro Manila", percentage: 89, sentiment: "Panic" },
    { name: "Batangas", percentage: 72, sentiment: "Fear/Anxiety" },
    { name: "Rizal", percentage: 65, sentiment: "Disbelief" },
    { name: "Laguna", percentage: 53, sentiment: "Fear/Anxiety" },
    { name: "Bulacan", percentage: 48, sentiment: "Resilience" }
  ];

  return (
    <div className="space-y-6">
      {/* Dashboard Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Dashboard</h1>
          <p className="mt-1 text-sm text-slate-500">Real-time disaster sentiment analysis overview</p>
        </div>
        <FileUploader 
          className="mt-4 sm:mt-0"
          onSuccess={() => {
            // The disaster context will handle refetching data
          }}
        />
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatusCard
          title="Active Disasters"
          value={activeDiastersCount}
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          iconBgColor="bg-blue-100"
          change={{ value: "+2 since yesterday", positive: true }}
        />

        <StatusCard
          title="Analyzed Posts"
          value={analyzedPostsCount}
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          iconBgColor="bg-indigo-100"
          change={{ value: `+${Math.floor(analyzedPostsCount * 0.15)} today`, positive: true }}
        />

        <StatusCard
          title="Dominant Sentiment"
          value={dominantSentiment}
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          iconBgColor="bg-amber-100"
          change={{ value: "Changed from Panic", positive: false }}
        />

        <StatusCard
          title="Model Confidence"
          value={`${Math.round(aiConfidence * 100)}%`}
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          iconBgColor="bg-green-100"
          change={{ value: "+5% improvement", positive: true }}
        />
      </div>

      {/* Sentiment Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sentiment Distribution Chart */}
        <div className="lg:col-span-2">
          <SentimentChart
            data={{
              labels: Object.keys(sentimentCounts),
              values: Object.values(sentimentCounts),
              title: "Sentiment Distribution",
              description: "Across all active disasters"
            }}
          />
        </div>

        {/* Top Affected Areas */}
        <AffectedAreas areas={affectedAreas} />
      </div>

      {/* Recent Posts Table */}
      <RecentPostsTable 
        posts={sentimentPosts}
        title="Recent Analyzed Posts"
        description="Latest social media sentiment"
      />
    </div>
  );
}
