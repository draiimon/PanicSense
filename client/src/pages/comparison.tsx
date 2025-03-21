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
    
    // Create a map of standard disaster types
    const standardDisasterTypes = {
      'earthquake': 'Earthquake',
      'quake': 'Earthquake',
      'lindol': 'Earthquake',
      'linog': 'Earthquake',
      'seismic': 'Earthquake',
      'tremor': 'Earthquake',
      'magnitude': 'Earthquake',
      
      'flood': 'Flood',
      'baha': 'Flood',
      'tubig': 'Flood',
      'inundation': 'Flood',
      'submerged': 'Flood',
      'overflow': 'Flood',
      
      'typhoon': 'Typhoon',
      'storm': 'Typhoon',
      'bagyo': 'Typhoon',
      'hurricane': 'Typhoon',
      'cyclone': 'Typhoon',
      'tropical': 'Typhoon',
      
      'fire': 'Fire',
      'sunog': 'Fire',
      'apoy': 'Fire',
      'blaze': 'Fire',
      'burning': 'Fire',
      'flames': 'Fire',
      
      'volcano': 'Volcano',
      'bulkan': 'Volcano',
      'eruption': 'Volcano',
      'lava': 'Volcano',
      'ash': 'Volcano',
      'lahar': 'Volcano',
      
      'landslide': 'Landslide',
      'mudslide': 'Landslide',
      'avalanche': 'Landslide',
      'guho': 'Landslide',
      'pagguho': 'Landslide',
      'rockslide': 'Landslide',
      
      'drought': 'Drought',
      'tagtuyot': 'Drought',
      'dry': 'Drought'
    };
    
    // Add standard disaster types from disasterEvents
    disasterEvents.forEach(event => {
      if (event.type && 
          event.type !== "Not Specified" && 
          event.type !== "null" && 
          event.type !== "undefined" &&
          event.type.toLowerCase() !== "none") {
        disasterTypeSet.add(event.type);
      }
    });
    
    // Add standardized disaster types from posts
    sentimentPosts.forEach(post => {
      if (!post.disasterType) return;
      if (post.disasterType === "Not Specified" || 
          post.disasterType === "NONE" || 
          post.disasterType === "None" || 
          post.disasterType === "null" || 
          post.disasterType === "undefined") {
        return;
      }
      
      // Try to standardize the disaster type
      const postDisasterType = post.disasterType.trim();
      const normalized = postDisasterType.toLowerCase();
      
      // Check if this is already a standardized type name (exact match)
      if (["Earthquake", "Flood", "Typhoon", "Fire", "Volcano", "Landslide", "Drought"].includes(postDisasterType)) {
        disasterTypeSet.add(postDisasterType);
      } else {
        // Look for keywords to standardize
        let matched = false;
        for (const [keyword, standardType] of Object.entries(standardDisasterTypes)) {
          if (normalized.includes(keyword)) {
            disasterTypeSet.add(standardType);
            matched = true;
            break;
          }
        }
        
        // If no match found but not a generic term, add as-is
        if (!matched && postDisasterType.length < 30) {
          disasterTypeSet.add(postDisasterType);
        }
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

  // Process data for disaster response effectiveness
  const processResponseEffectivenessData = () => {
    // Get all disaster types from posts
    const disasterTypes = new Set<string>();
    sentimentPosts.forEach(post => {
      if (post.disasterType) disasterTypes.add(post.disasterType);
    });

    // If no disaster types found, show default categories
    const categories = disasterTypes.size > 0 
      ? Array.from(disasterTypes) 
      : ['Typhoon', 'Flood', 'Earthquake', 'Fire'];

    // Calculate response effectiveness metrics for each disaster type
    const values = categories.map(disasterType => {
      // Get posts for this disaster type
      const postsForType = sentimentPosts.filter(post => 
        post.disasterType && post.disasterType.toLowerCase().includes(disasterType.toLowerCase())
      );
      
      if (postsForType.length === 0) return 75; // Default value if no posts

      // Sort by timestamp
      const sortedPosts = [...postsForType].sort((a, b) => 
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
      
      // Calculate percentage of positive sentiment (Resilience) in the later half
      // This shows how well response efforts worked over time
      const halfwayPoint = Math.floor(sortedPosts.length / 2);
      const laterPosts = sortedPosts.slice(halfwayPoint);
      
      if (laterPosts.length === 0) return 50;
      
      const positiveCount = laterPosts.filter(post => 
        post.sentiment === 'Resilience'
      ).length;
      
      // Calculate score: base 50% + percentage of positive sentiment (max 50%)
      const baseScore = 50;
      const positiveScore = laterPosts.length > 0 
        ? Math.round((positiveCount / laterPosts.length) * 50) 
        : 25;
      
      return baseScore + positiveScore;
    });

    return {
      labels: categories,
      values,
      title: "Disaster Response Effectiveness",
      description: "Higher values indicate better response effectiveness based on recovered sentiment"
    };
  };

  const responseEffectivenessData = processResponseEffectivenessData();

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
          data={responseEffectivenessData}
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
                // Check if sentiments array is empty
                if (disaster.sentiments.length === 0) {
                  return (
                    <li key={index} className="flex items-start space-x-3">
                      <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <p className="text-sm text-slate-700">
                        <span className="font-medium">{disaster.type}</span>: No sentiment data available.
                      </p>
                    </li>
                  );
                }
                
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