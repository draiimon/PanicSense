import { useDisasterContext } from "@/context/disaster-context";
import { FileUploader } from "@/components/file-uploader";
import { StatusCard } from "@/components/dashboard/status-card";
import { SentimentChart } from "@/components/dashboard/sentiment-chart";
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
    isLoadingSentimentPosts = false
  } = useDisasterContext();

  const sentimentData = {
    labels: ['Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral'],
    values: [0, 0, 0, 0, 0]
  };

  // Count sentiments
  sentimentPosts.forEach(post => {
    const index = sentimentData.labels.indexOf(post.sentiment);
    if (index !== -1) {
      sentimentData.values[index]++;
    }
  });

  return (
    <div className="p-6 max-w-[1600px] mx-auto">
      <motion.div 
        initial="hidden"
        animate="visible"
        variants={fadeInUp}
        className="mb-6"
      >
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-slate-800">Disaster Response Dashboard</h1>
            <p className="text-sm text-slate-500">Real-time sentiment monitoring and analysis</p>
          </div>
          <FileUploader className="mt-0" />
        </div>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <StatusCard 
          title="Active Disasters"
          value={activeDiastersCount}
          description="Currently monitored disasters"
        />
        <StatusCard 
          title="Analyzed Posts"
          value={analyzedPostsCount}
          description="Total posts processed"
        />
        <StatusCard 
          title="Dominant Sentiment"
          value={dominantSentiment}
          description="Most common sentiment"
        />
        <StatusCard 
          title="Model Confidence"
          value={`${(modelConfidence * 100).toFixed(1)}%`}
          description="Average prediction confidence"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="w-full">
          <SentimentChart data={sentimentData} />
        </div>
        <div className="w-full">
          <RecentPostsTable posts={sentimentPosts} limit={5} />
        </div>
      </div>
    </div>
  );
}