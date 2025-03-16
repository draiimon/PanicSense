import { useDisasterContext } from "@/context/disaster-context";
import { StatusCard } from "@/components/dashboard/status-card";
import { OptimizedSentimentChart } from "@/components/dashboard/optimized-sentiment-chart";
import { RecentPostsTable } from "@/components/dashboard/recent-posts-table";
import { AffectedAreasCard } from "@/components/dashboard/affected-areas-card";
import { FileUploader } from "@/components/file-uploader";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2 } from "lucide-react"; 

const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } }
};

function LoadingOverlay({ message }: { message: string }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center z-50">
      {/* Semi-transparent backdrop */}
      <div className="absolute inset-0 bg-white/90 backdrop-blur-lg"></div>

      {/* Loading content */}
      <div className="relative z-10 flex flex-col items-center gap-4 p-6 bg-white/50 rounded-xl shadow-lg backdrop-blur-sm">
        <Loader2 className="h-12 w-12 text-blue-600 animate-spin" />
        <div className="text-center">
          <p className="text-lg font-semibold text-slate-800">Processing Data</p>
          <p className="text-sm text-slate-600">{message}</p>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { 
    sentimentPosts = [],
    activeDiastersCount = 0,
    analyzedPostsCount = 0,
    dominantSentiment = 'N/A',
    modelConfidence = 0,
    isLoadingSentimentPosts = false
  } = useDisasterContext();

  // Filter posts from last week
  const lastWeekPosts = sentimentPosts.filter(post => {
    const postDate = new Date(post.timestamp);
    const weekAgo = new Date();
    weekAgo.setDate(weekAgo.getDate() - 7);
    return postDate >= weekAgo;
  });

  // Recalculate dominant sentiment from last week's posts
  const lastWeekDominantSentiment = (() => {
    if (lastWeekPosts.length === 0) return 'N/A';
    const sentimentCounts: Record<string, number> = {};
    lastWeekPosts.forEach(post => {
      sentimentCounts[post.sentiment] = (sentimentCounts[post.sentiment] || 0) + 1;
    });
    return Object.entries(sentimentCounts)
      .reduce((a, b) => a[1] > b[1] ? a : b)[0];
  })();

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
            {/* File uploader with loading state */}
            <div className="relative">
              <FileUploader className="mt-0" />
              {isLoadingSentimentPosts && (
                <div className="absolute inset-0 flex items-center justify-center bg-white/90 backdrop-blur-lg rounded-lg">
                  <Loader2 className="h-8 w-8 text-blue-600 animate-spin" />
                </div>
              )}
            </div>
          </CardHeader>
        </Card>
      </motion.div>

      {/* Status Cards */}
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
            className="relative"
          >
            <AffectedAreasCard 
              sentimentPosts={filteredPosts} 
              isLoading={isLoadingSentimentPosts}
            />
            {isLoadingSentimentPosts && (
              <LoadingOverlay message="Updating affected areas..." />
            )}
          </motion.div>
        </div>

        <div className="lg:col-span-2">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="grid grid-cols-1 gap-6 h-full"
          >
            <Card className="bg-white/50 backdrop-blur-sm border-none relative">
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

            <Card className="bg-white/50 backdrop-blur-sm border-none relative">
              <CardHeader>
                <CardTitle className="text-lg font-semibold">Recent Posts</CardTitle>
              </CardHeader>
              <CardContent>
                <RecentPostsTable 
                  posts={filteredPosts} 
                  limit={5}
                  isLoading={isLoadingSentimentPosts}
                />
                {isLoadingSentimentPosts && (
                  <LoadingOverlay message="Loading recent posts..." />
                )}
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
}