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
    <motion.div 
      initial="hidden"
      animate="visible"
      variants={{
        hidden: { opacity: 0 },
        visible: { opacity: 1 }
      }}
      className="h-[calc(100vh-2rem)] overflow-y-auto px-6 py-6 bg-slate-50"
    >
      <motion.div variants={fadeInUp} className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Disaster Response Dashboard</h1>
          <p className="text-sm text-slate-500">Real-time sentiment monitoring and analysis</p>
        </div>
        <FileUploader className="mt-0" />
      </motion.div>

      <motion.div variants={fadeInUp} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <StatusCard
          title="Active Disasters"
          value={activeDiastersCount}
          icon={<span className="text-2xl">🚨</span>}
          iconBgColor="bg-red-100"
        />
        <StatusCard
          title="Analyzed Posts"
          value={analyzedPostsCount}
          icon={<span className="text-2xl">📊</span>}
          iconBgColor="bg-blue-100"
        />
        <StatusCard
          title="Dominant Sentiment"
          value={dominantSentiment || "N/A"}
          icon={<span className="text-2xl">🎯</span>}
          iconBgColor="bg-purple-100"
        />
        <StatusCard
          title="Model Confidence"
          value={`${Math.round(modelConfidence * 100)}%`}
          icon={<span className="text-2xl">✨</span>}
          iconBgColor="bg-green-100"
        />
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <motion.div variants={fadeInUp} className="bg-white rounded-lg shadow-sm p-4">
          <h2 className="text-lg font-semibold mb-4">Sentiment Distribution</h2>
          <SentimentChart data={sentimentPercentages} />
        </motion.div>

        <motion.div variants={fadeInUp} className="bg-white rounded-lg shadow-sm p-4">
          <h2 className="text-lg font-semibold mb-4">Affected Areas</h2>
          <AffectedAreas />
        </motion.div>
      </div>

      <motion.div variants={fadeInUp} className="bg-white rounded-lg shadow-sm p-4">
        <h2 className="text-lg font-semibold mb-4">Recent Posts</h2>
        <RecentPostsTable posts={sentimentPosts.slice(0, 5)} />
      </motion.div>
    </motion.div>
  );
}