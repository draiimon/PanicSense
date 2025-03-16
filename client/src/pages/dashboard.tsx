import { useDisasterContext } from "@/context/disaster-context";
import { StatusCard } from "@/components/dashboard/status-card";
import { OptimizedSentimentChart } from "@/components/dashboard/optimized-sentiment-chart";
import { RecentPostsTable } from "@/components/dashboard/recent-posts-table";
import { AffectedAreasCard } from "@/components/dashboard/affected-areas-card";
import { FileUploader } from "@/components/file-uploader";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Loader2, Upload, Database, BarChart3, Globe2, ArrowUpRight, RefreshCw } from "lucide-react";
import { CardCarousel } from "@/components/dashboard/card-carousel";
import { Button } from "@/components/ui/button";

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
    disasterEvents = [],
    activeDiastersCount = 0,
    analyzedPostsCount = 0,
    dominantSentiment = 'N/A',
    modelConfidence = 0,
    isLoadingSentimentPosts = false
  } = useDisasterContext();

  // Calculate stats (added from changes)
  const totalPosts = sentimentPosts.length;
  const activeDisasters = disasterEvents.filter(event => 
    new Date(event.timestamp) >= new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
  ).length;

  // Get most affected area (with proper typing)
  const locationCounts = sentimentPosts.reduce<Record<string, number>>((acc, post) => {
    if (post.location) {
      acc[post.location] = (acc[post.location] || 0) + 1;
    }
    return acc;
  }, {});
  const mostAffectedArea = Object.entries(locationCounts)
    .sort(([,a], [,b]) => b - a)[0]?.[0] || 'N/A';


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
    <div className="space-y-8 pb-10">
      {/* Beautiful hero section with animated gradient and prominent upload button */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-700 shadow-xl"
      >
        <div className="absolute inset-0 bg-grid-white/10 bg-[size:20px_20px] opacity-20"></div>
        <div className="absolute h-40 w-40 rounded-full bg-blue-400 filter blur-3xl opacity-30 -top-20 -left-20 animate-pulse"></div>
        <div className="absolute h-40 w-40 rounded-full bg-indigo-400 filter blur-3xl opacity-30 -bottom-20 -right-20 animate-pulse"></div>
        
        <div className="relative px-6 py-12 sm:px-12 sm:py-16 flex flex-col md:flex-row items-center justify-between gap-8">
          <div className="md:max-w-[60%]">
            <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold text-white mb-4 leading-tight">
              Disaster Response Dashboard
            </h1>
            <p className="text-blue-100 text-base sm:text-lg mb-6 max-w-xl">
              Real-time sentiment monitoring and geospatial analysis for disaster response in the Philippines
            </p>
            
            <div className="flex flex-wrap gap-3">
              <div className="flex items-center text-xs bg-white/20 backdrop-blur-md px-4 py-2 rounded-full text-white">
                <Database className="h-3.5 w-3.5 mr-1.5" />
                <span>{totalPosts} Data Points</span>
              </div>
              <div className="flex items-center text-xs bg-white/20 backdrop-blur-md px-4 py-2 rounded-full text-white">
                <BarChart3 className="h-3.5 w-3.5 mr-1.5" />
                <span>Sentiment Analysis</span>
              </div>
              <div className="flex items-center text-xs bg-white/20 backdrop-blur-md px-4 py-2 rounded-full text-white">
                <Globe2 className="h-3.5 w-3.5 mr-1.5" />
                <span>Geographic Mapping</span>
              </div>
            </div>
          </div>
          
          {/* Prominent upload button with hover effect */}
          <div className="flex-shrink-0 relative min-w-[240px]">
            <motion.div
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.98 }}
              className="bg-white rounded-xl shadow-2xl overflow-hidden"
            >
              <div className="p-6">
                <div className="flex items-center mb-3">
                  <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
                    <Upload className="h-5 w-5 text-blue-600" />
                  </div>
                  <h3 className="ml-3 font-semibold text-gray-800">Upload Data</h3>
                </div>
                <p className="text-sm text-gray-600 mb-4">
                  Upload CSV files for instant sentiment analysis and disaster monitoring
                </p>
                <FileUploader className="w-full justify-center mt-2 !shadow-none hover:!bg-blue-700" />
              </div>
            </motion.div>
            {isLoadingSentimentPosts && (
              <div className="absolute inset-0 flex items-center justify-center bg-white/90 backdrop-blur-lg rounded-lg">
                <Loader2 className="h-8 w-8 text-blue-600 animate-spin" />
              </div>
            )}
          </div>
        </div>
      </motion.div>

      {/* Stats Grid with improved styling */}
      <motion.div 
        initial="hidden"
        animate="visible"
        variants={fadeInUp}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6"
      >
        <StatusCard 
          title="Active Disasters"
          value={activeDisasters.toString()}
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
          icon="bar-chart"
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

      {/* Flexbox layout for main content with improved proportions */}
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Left column */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="w-full lg:w-[450px] flex-shrink-0"
        >
          <div className="sticky top-6">
            <Card className="bg-white shadow-xl border-none overflow-hidden rounded-xl relative">
              <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-blue-100/50 pb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-lg bg-blue-500/10">
                      <Globe2 className="text-blue-600 h-5 w-5" />
                    </div>
                    <CardTitle className="text-lg font-semibold text-slate-800">Affected Areas</CardTitle>
                  </div>
                  <Button variant="ghost" size="sm" className="rounded-lg h-8 gap-1 text-xs font-medium text-blue-600 hover:text-blue-700 hover:bg-blue-50">
                    View All
                    <ArrowUpRight className="h-3 w-3" />
                  </Button>
                </div>
                <CardDescription className="text-slate-500 mt-1">
                  Recent disaster impact by location
                </CardDescription>
              </CardHeader>
              
              <div>
                <AffectedAreasCard 
                  sentimentPosts={filteredPosts} 
                  isLoading={isLoadingSentimentPosts}
                />
              </div>
              
              {isLoadingSentimentPosts && (
                <LoadingOverlay message="Updating affected areas..." />
              )}
            </Card>
          </div>
        </motion.div>

        {/* Right column - takes remaining space */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="flex-grow"
        >
          {/* Card Carousel for auto-rotating between Sentiment Distribution and Recent Activity */}
          <div className="relative mb-6 bg-white shadow-xl border-none rounded-xl overflow-hidden">
            <div className="absolute top-4 right-4 z-10 flex items-center gap-2">
              <span className="text-xs text-blue-600 font-medium flex items-center">
                <RefreshCw className="h-3 w-3 mr-1 animate-spin-slow" />
                Auto-rotating
              </span>
            </div>
            
            <CardCarousel 
              autoRotate={true}
              interval={10000}
              showControls={true}
              className="h-[450px]"
            >
              {/* Sentiment Distribution Card */}
              <div className="h-full">
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-blue-100/50 p-6 pb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="p-2 rounded-lg bg-blue-500/10">
                        <BarChart3 className="text-blue-600 h-5 w-5" />
                      </div>
                      <h3 className="text-lg font-semibold text-slate-800">Sentiment Distribution</h3>
                    </div>
                    <Button variant="ghost" size="sm" className="rounded-lg h-8 gap-1 text-xs font-medium text-blue-600 hover:text-blue-700 hover:bg-blue-50">
                      Full Analysis
                      <ArrowUpRight className="h-3 w-3" />
                    </Button>
                  </div>
                  <p className="text-sm text-slate-500 mt-1">
                    Emotional response breakdown across disaster events
                  </p>
                </div>
                <div className="p-6">
                  <div className="h-[350px]">
                    <OptimizedSentimentChart 
                      data={sentimentData}
                      isLoading={isLoadingSentimentPosts}
                    />
                  </div>
                </div>
              </div>

              {/* Recent Posts Card */}
              <div className="h-full flex flex-col">
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-blue-100/50 p-6 pb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="p-2 rounded-lg bg-blue-500/10">
                        <Database className="text-blue-600 h-5 w-5" />
                      </div>
                      <h3 className="text-lg font-semibold text-slate-800">Recent Activity</h3>
                    </div>
                    <Button variant="ghost" size="sm" className="rounded-lg h-8 gap-1 text-xs font-medium text-blue-600 hover:text-blue-700 hover:bg-blue-50">
                      View All
                      <ArrowUpRight className="h-3 w-3" />
                    </Button>
                  </div>
                  <p className="text-sm text-slate-500 mt-1">
                    Latest analyzed posts and sentiment data
                  </p>
                </div>
                <div className="flex-grow overflow-hidden relative">
                  <RecentPostsTable 
                    posts={filteredPosts} 
                    limit={5}
                    isLoading={isLoadingSentimentPosts}
                  />
                  {isLoadingSentimentPosts && (
                    <LoadingOverlay message="Loading recent posts..." />
                  )}
                </div>
              </div>
            </CardCarousel>
          </div>
        </motion.div>
      </div>
    </div>
  );
}