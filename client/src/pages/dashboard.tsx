
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
    sentimentPosts,
    activeDiastersCount,
    analyzedPostsCount,
    dominantSentiment,
    modelConfidence,
    isLoadingSentimentPosts
  } = useDisasterContext();

  const sentimentCounts = {
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

  const totalPosts = sentimentPosts.length || 1;
  const sentimentPercentages = Object.entries(sentimentCounts).map(([label, count]) => ({
    label,
    percentage: Math.round((count / totalPosts) * 100)
  }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
      <motion.div variants={fadeInUp} className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold text-slate-800">Disaster Response Dashboard</h1>
          <p className="text-sm text-slate-500">Real-time sentiment monitoring and analysis</p>
        </div>
        <FileUploader className="mt-0" />
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
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
          value={`${modelConfidence}%`}
          trend="+5%"
          trendDirection="up"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <SentimentChart data={sentimentPercentages} />
        <AffectedAreas />
      </div>

      <div className="grid grid-cols-1 gap-6">
        <RecentPostsTable
          data={sentimentPosts.slice(0, 5)}
          title="Recent Posts"
          description="Latest analyzed social media posts"
          showViewAllLink={true}
        />
      </div>
    </div>
  );
}
