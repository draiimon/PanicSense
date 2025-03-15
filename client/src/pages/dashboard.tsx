import { useDisasterContext } from "@/context/disaster-context";
import { StatusCard } from "@/components/dashboard/status-card";
import { OptimizedSentimentChart } from "@/components/dashboard/optimized-sentiment-chart";
import { RecentPostsTable } from "@/components/dashboard/recent-posts-table";
import { AffectedAreasCard } from "@/components/dashboard/affected-areas-card";
import { FileUploader } from "@/components/file-uploader";
import { motion, AnimatePresence } from "framer-motion";
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

  // Filter out "Not specified" and generic "Philippines" locations
  const filteredPosts = sentimentPosts.filter(post => {
    const location = post.location?.toLowerCase();
    return location && 
           location !== 'not specified' && 
           location !== 'philippines' &&
           location !== 'pilipinas' &&
           location !== 'pinas';
  });

  const sentimentData = {
    labels: ['Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral'],
    values: [0, 0, 0, 0, 0]
  };

  // Count sentiments from filtered posts
  filteredPosts.forEach(post => {
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
          isLoading={isLoadingSentimentPosts}
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
          isLoading={isLoadingSentimentPosts}
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
          isLoading={isLoadingSentimentPosts}
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
          isLoading={isLoadingSentimentPosts}
        />
      </motion.div>

      <motion.div 
        initial="hidden"
        animate="visible"
        variants={fadeInUp}
        className="grid grid-cols-1 lg:grid-cols-3 gap-6"
      >
        <div className="lg:col-span-1">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <AffectedAreasCard 
              sentimentPosts={filteredPosts} 
              isLoading={isLoadingSentimentPosts}
            />
          </motion.div>
        </div>

        <div className="lg:col-span-2">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="grid grid-cols-1 gap-6 h-full"
          >
            <Card className="bg-white/50 backdrop-blur-sm border-none">
              <CardHeader>
                <CardTitle className="text-lg font-semibold">Sentiment Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <OptimizedSentimentChart 
                  data={sentimentData}
                  isLoading={isLoadingSentimentPosts}
                />
              </CardContent>
            </Card>

            <Card className="bg-white/50 backdrop-blur-sm border-none">
              <CardHeader>
                <CardTitle className="text-lg font-semibold">Recent Posts</CardTitle>
              </CardHeader>
              <CardContent>
                <RecentPostsTable 
                  posts={filteredPosts} 
                  limit={5}
                  isLoading={isLoadingSentimentPosts}
                />
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
}