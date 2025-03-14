import { useState } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { DisasterComparison } from "@/components/comparison/disaster-comparison";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { SentimentChart } from "@/components/dashboard/sentiment-chart";

export default function Comparison() {
  const { sentimentPosts, disasterEvents } = useDisasterContext();

  // Process sentiment data by disaster type
  const processDisasterData = () => {
    // Get unique disaster types, but filter out "Not Specified", "Not mentioned", "Unspecified" etc.
    const disasterTypeSet = new Set<string>();
    
    // Manual collection to avoid TypeScript issues with Set
    sentimentPosts.forEach(post => {
      if (post.disasterType && post.disasterType !== "Not Specified") {
        // Normalize disaster type names to ensure consistency
        let disasterType = post.disasterType;
        
        // Handle common variations to standardize
        const normalized = disasterType.toLowerCase().trim();
        if (normalized.includes("earthquake") || normalized.includes("quake") || normalized.includes("lindol")) {
          disasterType = "Earthquake";
        } else if (normalized.includes("flood") || normalized.includes("baha")) {
          disasterType = "Flood";
        } else if (normalized.includes("typhoon") || normalized.includes("storm") || normalized.includes("bagyo")) {
          disasterType = "Typhoon";
        } else if (normalized.includes("fire") || normalized.includes("sunog")) {
          disasterType = "Fire";
        } else if (normalized.includes("volcano") || normalized.includes("eruption") || normalized.includes("bulkan")) {
          disasterType = "Volcano";
        } else if (normalized.includes("landslide") || normalized.includes("mudslide") || normalized.includes("guho")) {
          disasterType = "Landslide";
        } else if (normalized.includes("drought") || normalized.includes("tagtuyot")) {
          disasterType = "Drought";
        }
        
        disasterTypeSet.add(disasterType);
      }
    });
    
    // Filter out generic unknown values for more professional presentation
    const validDisasterTypes = Array.from(disasterTypeSet).filter(type => {
      if (!type) return false;
      
      const lowerType = type.toLowerCase();
      
      // Filter out placeholder values and long phrases
      return !(
        lowerType === "not specified" || 
        lowerType === "not mentioned" || 
        lowerType === "unspecified" || 
        lowerType === "none" ||
        lowerType.includes("none specifically mentioned") ||
        lowerType.includes("but it implies") ||
        type.length > 40  // Filter out very long descriptions which are likely placeholders
      );
    });

    return validDisasterTypes.map(type => {
      // Find posts for this disaster type including partial matches
      const postsForType = sentimentPosts.filter(post => {
        if (!post.disasterType) return false;
        
        // Handle exact matches
        if (post.disasterType === type) return true;
        
        // Handle partial/similar matches based on keywords
        const disasterLower = post.disasterType.toLowerCase();
        const typeLower = type.toLowerCase();
        
        // Check if the disaster type is part of the post disaster type or has similar keywords
        if (disasterLower.includes(typeLower)) return true;
        
        // Check specific keyword matches by disaster type
        switch (type) {
          case "Earthquake":
            return disasterLower.includes("quake") || disasterLower.includes("lindol") || disasterLower.includes("linog");
          case "Flood":
            return disasterLower.includes("baha") || disasterLower.includes("tubig") || disasterLower.includes("inundation");
          case "Typhoon":
            return disasterLower.includes("bagyo") || disasterLower.includes("storm") || disasterLower.includes("hurricane");
          case "Fire":
            return disasterLower.includes("sunog") || disasterLower.includes("apoy") || disasterLower.includes("burning");
          case "Volcano":
            return disasterLower.includes("bulkan") || disasterLower.includes("eruption") || disasterLower.includes("lahar");
          case "Landslide":
            return disasterLower.includes("guho") || disasterLower.includes("mudslide") || disasterLower.includes("avalanche");
          default:
            return false;
        }
      });
      
      const totalPosts = postsForType.length;

      // Fix TypeScript type issues with sentiment counts
      const sentimentCounts: Record<string, number> = {};
      postsForType.forEach(post => {
        sentimentCounts[post.sentiment] = (sentimentCounts[post.sentiment] || 0) + 1;
      });

      const sentiments = Object.entries(sentimentCounts).map(([label, count]) => ({
        label,
        percentage: totalPosts > 0 ? (count / totalPosts) * 100 : 0
      }));

      return {
        type,
        sentiments
      };
    });
  };

  const disasterData = processDisasterData();

  // Process comparison data for phases
  const processPhaseData = () => {
    // Sort posts by timestamp to analyze phases
    const sortedPosts = [...sentimentPosts].sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    // Split posts into three phases
    const totalPosts = sortedPosts.length;
    const postsPerPhase = Math.ceil(totalPosts / 3);

    const phases = ['Initial Phase', 'Peak Phase', 'Recovery Phase'];
    const values = phases.map((_, index) => {
      const start = index * postsPerPhase;
      const end = Math.min(start + postsPerPhase, totalPosts);
      const phasePosts = sortedPosts.slice(start, end);

      // Calculate intensity based on negative sentiments
      const negativeCount = phasePosts.filter(post => 
        ['Panic', 'Fear/Anxiety', 'Disbelief'].includes(post.sentiment)
      ).length;

      return phasePosts.length > 0 ? (negativeCount / phasePosts.length) * 100 : 0;
    });

    return {
      labels: phases,
      values,
      title: "Sentiment Intensity by Disaster Phase",
      description: "How emotions evolve throughout disaster lifecycle"
    };
  };

  const timeComparisonData = processPhaseData();

  return (
    <div className="space-y-6">
      {/* Comparison Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Comparison Analysis</h1>
        <p className="mt-1 text-sm text-slate-500">Analyzing sentiment distribution across different disasters</p>
      </div>

      {/* Disaster Comparison Chart */}
      <DisasterComparison 
        disasters={disasterData}
        title="Disaster Type Comparison"
        description="Sentiment distribution across different disasters"
      />

      {/* Additional comparison insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <SentimentChart 
          data={timeComparisonData}
          type="bar"
        />

        <Card className="bg-white rounded-lg shadow">
          <CardHeader className="p-5 border-b border-gray-200">
            <CardTitle className="text-lg font-medium text-slate-800">Analysis Insights</CardTitle>
            <CardDescription className="text-sm text-slate-500">
              Key observations from cross-disaster analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="p-5">
            <ul className="space-y-4">
              {disasterData.map((disaster, index) => {
                const dominantSentiment = disaster.sentiments.reduce((prev, current) => 
                  current.percentage > prev.percentage ? current : prev
                );

                return (
                  <li key={index} className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <p className="text-sm text-slate-700">
                      <span className="font-medium">{disaster.type}</span>: Predominantly shows {dominantSentiment.label.toLowerCase()} sentiment ({dominantSentiment.percentage.toFixed(1)}%), 
                      indicating {
                        dominantSentiment.label === 'Resilience' ? 'strong community response and adaptation' :
                        dominantSentiment.label === 'Panic' ? 'immediate distress and urgent needs' :
                        dominantSentiment.label === 'Fear/Anxiety' ? 'ongoing concern and uncertainty' :
                        dominantSentiment.label === 'Disbelief' ? 'shock and difficulty accepting the situation' :
                        'neutral information sharing'
                      }.
                    </p>
                  </li>
                );
              })}
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}