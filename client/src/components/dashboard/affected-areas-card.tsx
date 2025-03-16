import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle, 
  CardDescription 
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getSentimentBadgeClasses } from "@/lib/colors";
import { getDisasterTypeColor } from "@/lib/colors";
import { MapPin, AlertTriangle, TrendingUp } from "lucide-react";
import { SentimentPost } from "@/lib/api";

interface AffectedAreaProps {
  sentimentPosts: SentimentPost[];
}

interface AffectedArea {
  name: string;
  sentiment: string;
  disasterType: string | null;
  impactLevel: number;
}

export function AffectedAreasCard({ sentimentPosts }: AffectedAreaProps) {
  const [affectedAreas, setAffectedAreas] = useState<AffectedArea[]>([]);

  useEffect(() => {
    // Extract and count location mentions
    const locationCount = new Map<string, { 
      count: number,
      sentiment: Map<string, number>,
      disasterType: Map<string, number>
    }>();

    sentimentPosts.forEach(post => {
      if (!post.location) return;

      const location = post.location;

      if (!locationCount.has(location)) {
        locationCount.set(location, {
          count: 0,
          sentiment: new Map(),
          disasterType: new Map()
        });
      }

      const locationData = locationCount.get(location)!;
      locationData.count++;

      // Track sentiments
      const currentSentimentCount = locationData.sentiment.get(post.sentiment) || 0;
      locationData.sentiment.set(post.sentiment, currentSentimentCount + 1);

      // Track disaster types
      if (post.disasterType) {
        const currentTypeCount = locationData.disasterType.get(post.disasterType) || 0;
        locationData.disasterType.set(post.disasterType, currentTypeCount + 1);
      }
    });

    // Convert to array and sort by count
    const sortedAreas = Array.from(locationCount.entries())
      .map(([name, data]) => {
        // Get dominant sentiment
        let maxSentimentCount = 0;
        let dominantSentiment = "Neutral";

        data.sentiment.forEach((count, sentiment) => {
          if (count > maxSentimentCount) {
            maxSentimentCount = count;
            dominantSentiment = sentiment;
          }
        });

        // Get dominant disaster type
        let maxTypeCount = 0;
        let dominantType: string | null = null;

        data.disasterType.forEach((count, type) => {
          if (count > maxTypeCount) {
            maxTypeCount = count;
            dominantType = type;
          }
        });

        return {
          name,
          sentiment: dominantSentiment,
          disasterType: dominantType,
          impactLevel: data.count
        };
      })
      .sort((a, b) => b.impactLevel - a.impactLevel)
      .slice(0, 10); // Modified to show top 10

    setAffectedAreas(sortedAreas);
  }, [sentimentPosts]);

  return (
    <Card className="bg-white/50 backdrop-blur-sm border-none h-[800px] overflow-y-auto"> {/* Modified card height */}
      <CardHeader>
        <div className="flex items-center gap-2">
          <MapPin className="text-red-500 h-5 w-5" />
          <CardTitle className="text-lg font-semibold">Recent Affected Areas</CardTitle>
        </div>
        <CardDescription>Locations with recent disaster mentions</CardDescription>
      </CardHeader>
      <CardContent>
        <AnimatePresence>
          <div className="space-y-4">
            {affectedAreas.length === 0 ? (
              <p className="text-center text-sm text-slate-500">No affected areas detected</p>
            ) : (
              affectedAreas.map((area, index) => (
                <motion.div
                  key={area.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ 
                    opacity: 1, 
                    y: 0, 
                    transition: { 
                      delay: index * 0.1,
                      duration: 0.5 
                    } 
                  }}
                  className="rounded-lg p-3 bg-white border border-gray-100 shadow-sm"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-2">
                      <div className="mt-0.5">
                        <AlertTriangle 
                          className="h-4 w-4" 
                          style={{ 
                            color: area.disasterType ? 
                              getDisasterTypeColor(area.disasterType) : 
                              "#6b7280" 
                          }} 
                        />
                      </div>
                      <div>
                        <h3 className="font-medium text-gray-900">{area.name}</h3>
                        <div className="flex flex-wrap gap-1 mt-1">
                          <Badge 
                            className={getSentimentBadgeClasses(area.sentiment)}
                          >
                            {area.sentiment}
                          </Badge>

                          {area.disasterType && (
                            <Badge
                              style={{
                                backgroundColor: getDisasterTypeColor(area.disasterType),
                                color: 'white'
                              }}
                            >
                              {area.disasterType}
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-1">
                      <TrendingUp className="h-3 w-3 text-amber-500" />
                      <span className="text-xs font-medium text-amber-600">
                        {area.impactLevel} {area.impactLevel === 1 ? 'mention' : 'mentions'}
                      </span>
                    </div>
                  </div>
                </motion.div>
              ))
            )}
          </div>
        </AnimatePresence>
      </CardContent>
    </Card>
  );
}