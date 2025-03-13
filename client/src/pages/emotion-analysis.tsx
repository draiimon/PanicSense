import { useState } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { SentimentMap } from "@/components/analysis/sentiment-map";
import { SentimentLegend } from "@/components/analysis/sentiment-legend";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export default function EmotionAnalysis() {
  const { sentimentPosts, disasterEvents } = useDisasterContext();
  const [selectedDisasterType, setSelectedDisasterType] = useState<string>("All Disaster Types");
  const [selectedRegion, setSelectedRegion] = useState<{
    name: string;
    sentiments: { name: string; percentage: number }[];
  } | undefined>(undefined);

  // Create a list of unique disaster types
  const disasterTypes = ["All Disaster Types", ...new Set(disasterEvents.map(event => event.type))];

  // Mock regions data for the map
  // In a real app, this would be derived from sentimentPosts with location data
  const regions = [
    {
      name: "Metro Manila",
      coordinates: [14.5995, 120.9842],
      sentiment: "Panic",
      intensity: 89
    },
    {
      name: "Batangas",
      coordinates: [13.7565, 121.0583],
      sentiment: "Fear/Anxiety",
      intensity: 72
    },
    {
      name: "Rizal",
      coordinates: [14.6042, 121.3035],
      sentiment: "Disbelief",
      intensity: 65
    },
    {
      name: "Laguna",
      coordinates: [14.2691, 121.4113],
      sentiment: "Fear/Anxiety",
      intensity: 53
    },
    {
      name: "Bulacan",
      coordinates: [14.7969, 120.8787],
      sentiment: "Resilience",
      intensity: 48
    }
  ];

  // Mock most affected areas
  const mostAffectedAreas = [
    { name: "Metro Manila", sentiment: "Panic" },
    { name: "Batangas", sentiment: "Fear/Anxiety" },
    { name: "Rizal", sentiment: "Disbelief" }
  ];

  const handleRegionSelect = (region: any) => {
    // In a real app, sentiment distribution would be calculated based on actual data
    setSelectedRegion({
      name: region.name,
      sentiments: [
        { name: "Panic", percentage: region.sentiment === "Panic" ? 65 : 15 },
        { name: "Fear/Anxiety", percentage: region.sentiment === "Fear/Anxiety" ? 45 : 20 },
        { name: "Disbelief", percentage: region.sentiment === "Disbelief" ? 55 : 25 },
        { name: "Resilience", percentage: region.sentiment === "Resilience" ? 40 : 10 },
        { name: "Neutral", percentage: 10 }
      ]
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
