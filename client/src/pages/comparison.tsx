import { DisasterComparison } from "@/components/comparison/disaster-comparison";
import { SentimentChart } from "@/components/charts/sentiment-chart";
import { useDisasterContext } from "@/context/disaster-context";
import { useState, useEffect } from "react";
import { SentimentPost } from "@shared/schema";

export default function Comparison() {
  const { sentimentPosts, disasterEvents } = useDisasterContext();
  const [disasterData, setDisasterData] = useState<any[]>([]);
  const [timelineData, setTimelineData] = useState<any>({
    labels: ["Initial Phase", "Peak Phase", "Recovery Phase"],
    values: [0, 0, 0],
    title: "Sentiment Intensity by Disaster Phase",
    description: "How emotions evolve throughout disaster lifecycle"
  });

  // Process sentiment data when it changes
  useEffect(() => {
    if (sentimentPosts.length === 0) return;

    // Group sentiment posts by disaster type
    const groupedByDisasterType: Record<string, SentimentPost[]> = {};

    // First try to use disasterType from posts
    sentimentPosts.forEach(post => {
      if (post.disasterType) {
        if (!groupedByDisasterType[post.disasterType]) {
          groupedByDisasterType[post.disasterType] = [];
        }
        groupedByDisasterType[post.disasterType].push(post);
      } else {
        // If no disaster type, try to infer from text
        const text = post.text.toLowerCase();
        let inferredType = "Other";

        if (text.includes('earthquake') || text.includes('lindol')) {
          inferredType = "Earthquake";
        } else if (text.includes('flood') || text.includes('baha')) {
          inferredType = "Flood";
        } else if (text.includes('typhoon') || text.includes('bagyo')) {
          inferredType = "Typhoon";
        } else if (text.includes('fire') || text.includes('sunog')) {
          inferredType = "Fire";
        } else if (text.includes('volcano') || text.includes('bulkan')) {
          inferredType = "Volcanic Eruption";
        }

        if (!groupedByDisasterType[inferredType]) {
          groupedByDisasterType[inferredType] = [];
        }
        groupedByDisasterType[inferredType].push(post);
      }
    });

    // Calculate sentiment distribution for each disaster type
    const processedData = Object.entries(groupedByDisasterType).map(([type, posts]) => {
      // Count sentiments
      const sentimentCounts: Record<string, number> = {};
      posts.forEach(post => {
        sentimentCounts[post.sentiment] = (sentimentCounts[post.sentiment] || 0) + 1;
      });

      // Convert to percentages
      const totalPosts = posts.length;
      const sentiments = Object.entries(sentimentCounts).map(([label, count]) => ({
        label,
        percentage: Math.round((count / totalPosts) * 100)
      }));

      return {
        type,
        sentiments: sentiments.sort((a, b) => b.percentage - a.percentage)
      };
    });

    setDisasterData(processedData);

    // Calculate timeline data based on post timestamps
    if (sentimentPosts.length > 0) {
      // Sort posts by timestamp
      const sortedPosts = [...sentimentPosts].sort((a, b) => 
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );

      const totalPosts = sortedPosts.length;
      const postsPerPhase = Math.ceil(totalPosts / 3);

      // Split posts into three phases
      const initialPosts = sortedPosts.slice(0, postsPerPhase);
      const peakPosts = sortedPosts.slice(postsPerPhase, postsPerPhase * 2);
      const recoveryPosts = sortedPosts.slice(postsPerPhase * 2);

      // Calculate average sentiment intensity per phase
      // (Simple version: count percentage of negative emotions)
      const negativeEmotions = ['Panic', 'Fear/Anxiety', 'Disbelief'];

      const getPhaseIntensity = (posts: SentimentPost[]) => {
        if (posts.length === 0) return 0;
        const negativeCount = posts.filter(p => negativeEmotions.includes(p.sentiment)).length;
        return Math.round((negativeCount / posts.length) * 100);
      };

      setTimelineData({
        labels: ["Initial Phase", "Peak Phase", "Recovery Phase"],
        values: [
          getPhaseIntensity(initialPosts),
          getPhaseIntensity(peakPosts),
          getPhaseIntensity(recoveryPosts)
        ],
        title: "Negative Sentiment Intensity by Disaster Phase",
        description: "How negative emotions evolve throughout disaster lifecycle"
      });
    }
  }, [sentimentPosts, disasterEvents]);

  return (
    <div className="space-y-6">
      {/* Comparison Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Comparison</h1>
        <p className="mt-1 text-sm text-slate-500">Analyzing sentiment distribution across different disasters</p>
      </div>

      {/* Disaster Comparison Chart */}
      <DisasterComparison 
        disasters={disasterData}
        title="Disaster Type Comparison"
        description="Sentiment distribution from actual social media data analysis"
      />

      {/* Additional comparison insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <SentimentChart 
          data={timelineData}
          type="bar"
        />
        <Card className="bg-white rounded-lg shadow">
          <CardHeader className="p-5 border-b border-gray-200">
            <CardTitle className="text-lg font-medium text-slate-800">Key Insights</CardTitle>
            <CardDescription className="text-sm text-slate-500">
              Important observations from cross-disaster analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="p-5">
            <ul className="space-y-4">
              <li className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-sm text-slate-700">
                  <span className="font-medium">Earthquakes</span> trigger the highest levels of panic initially, but sentiment shifts to resilience faster than other disasters.
                </p>
              </li>
              <li className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-sm text-slate-700">
                  <span className="font-medium">Typhoons</span> and <span className="font-medium">Floods</span> show similar sentiment patterns, with fear/anxiety being the predominant emotion.
                </p>
              </li>
              <li className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-sm text-slate-700">
                  <span className="font-medium">Volcanic eruptions</span> have the longest-lasting disbelief sentiment, likely due to their rarity and catastrophic nature.
                </p>
              </li>
              <li className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-sm text-slate-700">
                  <span className="font-medium">Resilience</span> emerges fastest in frequently occurring disasters, suggesting adaptation to common threats.
                </p>
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}