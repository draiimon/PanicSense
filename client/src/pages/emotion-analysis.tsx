import { useState } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { SentimentMap } from "@/components/analysis/sentiment-map";
import { SentimentLegend } from "@/components/analysis/sentiment-legend";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useMemo } from "react";

export default function EmotionAnalysis() {
  const { sentimentPosts, disasterEvents } = useDisasterContext();
  const [selectedDisasterType, setSelectedDisasterType] = useState<string>("All Disaster Types");
  const [selectedRegion, setSelectedRegion] = useState<{
    name: string;
    sentiments: { name: string; percentage: number }[];
  } | undefined>(undefined);

  // Create a list of unique disaster types
  const disasterTypes = ["All Disaster Types", ...new Set(disasterEvents.map(event => event.type))];

  // Process real data for regions
  const regions = useMemo(() => {
    const locationData = new Map<string, {
      count: number;
      sentiments: Map<string, number>;
      coordinates: [number, number];
    }>();

    // Philippine region coordinates (you can extend this map)
    const regionCoordinates: Record<string, [number, number]> = {
      "Metro Manila": [14.5995, 120.9842],
      "Batangas": [13.7565, 121.0583],
      "Rizal": [14.6042, 121.3035],
      "Laguna": [14.2691, 121.4113],
      "Bulacan": [14.7969, 120.8787],
      "Cavite": [14.4791, 120.8970],
      "Cebu": [10.3157, 123.8854],
      "Davao": [7.0707, 125.6087],
      "Pampanga": [15.0794, 120.6200]
    };

    // Process posts to gather location data
    sentimentPosts.forEach(post => {
      if (!post.location || !regionCoordinates[post.location]) return;

      if (!locationData.has(post.location)) {
        locationData.set(post.location, {
          count: 0,
          sentiments: new Map(),
          coordinates: regionCoordinates[post.location]
        });
      }

      const data = locationData.get(post.location)!;
      data.count++;

      const currentSentimentCount = data.sentiments.get(post.sentiment) || 0;
      data.sentiments.set(post.sentiment, currentSentimentCount + 1);
    });

    // Convert to array and calculate dominant sentiments and intensities
    return Array.from(locationData.entries()).map(([name, data]) => {
      // Find dominant sentiment
      let maxCount = 0;
      let dominantSentiment = "Neutral";

      data.sentiments.forEach((count, sentiment) => {
        if (count > maxCount) {
          maxCount = count;
          dominantSentiment = sentiment;
        }
      });

      // Calculate intensity based on post count relative to maximum
      const maxPosts = Math.max(...Array.from(locationData.values()).map(d => d.count));
      const intensity = (data.count / maxPosts) * 100;

      return {
        name,
        coordinates: data.coordinates,
        sentiment: dominantSentiment,
        intensity
      };
    });
  }, [sentimentPosts]);

  // Calculate most affected areas
  const mostAffectedAreas = useMemo(() => {
    return regions
      .sort((a, b) => b.intensity - a.intensity)
      .slice(0, 3)
      .map(region => ({
        name: region.name,
        sentiment: region.sentiment
      }));
  }, [regions]);

  const handleRegionSelect = (region: any) => {
    const regionData = locationData.get(region.name);
    if (!regionData) return;

    const totalSentiments = Array.from(regionData.sentiments.values()).reduce((sum, count) => sum + count, 0);

    setSelectedRegion({
      name: region.name,
      sentiments: Array.from(regionData.sentiments.entries()).map(([name, count]) => ({
        name,
        percentage: (count / totalSentiments) * 100
      }))
    });
  };

  return (
    <div className="space-y-6">
      {/* Emotion Analysis Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Emotion Analysis</h1>
          <p className="mt-1 text-sm text-slate-500">Mapping emotions by geographic area</p>
        </div>
        <div className="mt-4 sm:mt-0">
          <Select
            value={selectedDisasterType}
            onValueChange={setSelectedDisasterType}
          >
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="All Disaster Types" />
            </SelectTrigger>
            <SelectContent>
              {disasterTypes.map((type) => (
                <SelectItem key={type} value={type}>
                  {type}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Map and Legend */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sentiment Map */}
        <div className="lg:col-span-2">
          <SentimentMap 
            regions={regions}
            onRegionSelect={handleRegionSelect}
          />
        </div>

        {/* Sentiment Legend and Stats */}
        <SentimentLegend 
          selectedRegion={selectedRegion}
          mostAffectedAreas={mostAffectedAreas}
        />
      </div>
    </div>
  );
}
import { useState, useEffect } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export default function EmotionAnalysis() {
  const { sentimentPosts, disasterEvents } = useDisasterContext();
  const [selectedDisasterType, setSelectedDisasterType] = useState<string>("All");
  const [affectedAreas, setAffectedAreas] = useState<{name: string, count: number}[]>([]);

  // Process affected areas
  useEffect(() => {
    const areaCount = new Map<string, number>();
    sentimentPosts.forEach(post => {
      if (post.location) {
        areaCount.set(post.location, (areaCount.get(post.location) || 0) + 1);
      }
    });

    const sortedAreas = Array.from(areaCount.entries())
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    setAffectedAreas(sortedAreas);
  }, [sentimentPosts]);

  // Filter posts by disaster type
  const filteredPosts = selectedDisasterType === "All" 
    ? sentimentPosts
    : sentimentPosts.filter(post => post.disasterType === selectedDisasterType);

  // Get unique disaster types
  const disasterTypes = ["All", ...new Set(sentimentPosts.map(post => post.disasterType).filter(Boolean))];

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Emotional Analysis</h1>
        <Select value={selectedDisasterType} onValueChange={setSelectedDisasterType}>
          <SelectTrigger className="w-[200px]">
            <SelectValue placeholder="Filter by disaster type" />
          </SelectTrigger>
          <SelectContent>
            {disasterTypes.map(type => (
              <SelectItem key={type} value={type}>{type}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Most Affected Areas</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {affectedAreas.map(area => (
                <div key={area.name} className="flex justify-between items-center">
                  <span>{area.name}</span>
                  <span className="text-sm text-muted-foreground">{area.count} reports</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Emotional Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {['Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral'].map(emotion => {
                const count = filteredPosts.filter(post => post.sentiment === emotion).length;
                const percentage = (count / filteredPosts.length) * 100 || 0;
                return (
                  <div key={emotion} className="flex flex-col">
                    <div className="flex justify-between mb-1">
                      <span>{emotion}</span>
                      <span>{percentage.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent Reports</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredPosts.slice(0, 5).map(post => (
              <div key={post.id} className="p-4 border rounded-lg">
                <div className="flex justify-between mb-2">
                  <span className="font-medium">{post.disasterType || 'Unknown Disaster'}</span>
                  <span className="text-sm text-muted-foreground">
                    {new Date(post.timestamp).toLocaleDateString()}
                  </span>
                </div>
                <p className="text-sm mb-2">{post.text}</p>
                <div className="flex gap-2">
                  <span className="text-xs px-2 py-1 bg-blue-100 rounded">
                    {post.sentiment}
                  </span>
                  {post.location && (
                    <span className="text-xs px-2 py-1 bg-gray-100 rounded">
                      {post.location}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
