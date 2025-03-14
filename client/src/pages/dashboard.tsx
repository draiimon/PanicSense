
import { useDisasterContext } from "@/context/disaster-context";
import { FileUploader } from "@/components/file-uploader";
import { StatusCard } from "@/components/dashboard/status-card";
import { SentimentChart } from "@/components/dashboard/sentiment-chart";
import { AffectedAreas } from "@/components/dashboard/affected-areas";
import { RecentPostsTable } from "@/components/dashboard/recent-posts-table";
import { motion } from "framer-motion";

const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } }
};

export default function Dashboard() {
  const { 
    sentimentPosts = [],
    activeDiastersCount = 0,
    analyzedPostsCount = 0,
    dominantSentiment = 'N/A',
    modelConfidence = 0,
  } = useDisasterContext();

  // Calculate sentiment percentages
  const sentimentCounts = sentimentPosts.reduce((acc, post) => {
    acc[post.sentiment] = (acc[post.sentiment] || 0) + 1;
    return acc;
  }, {
    'Panic': 0,
    'Fear/Anxiety': 0,
    'Disbelief': 0,
    'Resilience': 0,
    'Neutral': 0
  });

  const totalPosts = Math.max(1, sentimentPosts.length);
  const sentimentPercentages = Object.entries(sentimentCounts).map(([label, count]) => ({
    label,
    percentage: Math.round((count as number / totalPosts) * 100)
  }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="max-w-[2000px] mx-auto px-6 py-8 space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-slate-800">Disaster Response Dashboard</h1>
            <p className="text-sm text-slate-500">Real-time sentiment monitoring and analysis</p>
          </div>
          <FileUploader className="mt-0" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatusCard
            title="Active Disasters"
            value={activeDiastersCount}
            trend="+2"
            trendDirection="up"
          />
          <StatusCard
            title="Analyzed Posts"
            value={analyzedPostsCount}
            trend="+120"
            trendDirection="up"
          />
          <StatusCard
            title="Dominant Sentiment"
            value={dominantSentiment}
            trend="stable"
            trendDirection="neutral"
          />
          <StatusCard
            title="Model Confidence"
            value={`${Math.round(modelConfidence * 100)}%`}
            trend="+5%"
            trendDirection="up"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-lg font-semibold mb-4">Sentiment Distribution</h2>
            <SentimentChart data={sentimentPercentages} />
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-lg font-semibold mb-4">Affected Areas</h2>
            <AffectedAreas />
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold mb-4">Recent Posts</h2>
          <RecentPostsTable 
            data={sentimentPosts || []}
            title="Recent Posts"
            description="Latest analyzed social media posts"
            showViewAllLink={true}
            limit={5}
          />
        </div>
      </div>
    </div>
  );
}
