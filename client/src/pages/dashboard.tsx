import { useDisasterContext } from "@/context/disaster-context";
import { StatusCard } from "@/components/dashboard/status-card";
import { SentimentChart } from "@/components/dashboard/sentiment-chart";
import { RecentPostsTable } from "@/components/dashboard/recent-posts-table";
import { FileUploader } from "@/components/file-uploader";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

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
    <div className="space-y-8">
      <motion.div 
        initial="hidden"
        animate="visible"
        variants={fadeInUp}
      >
        <Card className="bg-white/50 backdrop-blur-sm border-none shadow-md">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-7">
            <div>
              <CardTitle className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Disaster Response Dashboard
              </CardTitle>
              <p className="text-sm text-slate-500 mt-1">Real-time sentiment monitoring and analysis</p>
            </div>
            <FileUploader className="mt-0" />
          </CardHeader>
        </Card>
      </motion.div>

      <motion.div 
        initial="hidden"
        animate="visible"
        variants={fadeInUp}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
      >
        <StatusCard 
          title="Active Disasters"
          value={activeDiastersCount.toString()}
          icon="alert-triangle"
          trend={{
            value: "+2",
            isUpward: true,
            label: "from last week"
          }}
        />
        <StatusCard 
          title="Analyzed Posts"
          value={analyzedPostsCount.toString()}
          icon="bar-chart"
          trend={{
            value: "+15%",
            isUpward: true,
            label: "increase in analysis"
          }}
        />
        <StatusCard 
          title="Dominant Sentiment"
          value={dominantSentiment}
          icon="brain"
          trend={{
            value: "stable",
            isUpward: null,
            label: "sentiment trend"
          }}
        />
        <StatusCard 
          title="Model Confidence"
          value={`${(modelConfidence * 100).toFixed(1)}%`}
          icon="check-circle"
          trend={{
            value: "+5%",
            isUpward: true,
            label: "accuracy improvement"
          }}
        />
      </motion.div>

      <motion.div 
        initial="hidden"
        animate="visible"
        variants={fadeInUp}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        <Card className="bg-white/50 backdrop-blur-sm border-none">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Sentiment Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <SentimentChart data={sentimentData} />
          </CardContent>
        </Card>

        <Card className="bg-white/50 backdrop-blur-sm border-none">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Recent Posts</CardTitle>
          </CardHeader>
          <CardContent>
            <RecentPostsTable posts={sentimentPosts} limit={5} />
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}