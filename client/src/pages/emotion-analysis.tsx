import { useState, useMemo } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { SentimentMap } from "@/components/analysis/sentiment-map";
import { SentimentLegend } from "@/components/analysis/sentiment-legend";

export default function EmotionAnalysis() {
  const { sentimentPosts } = useDisasterContext();
  const [selectedRegion, setSelectedRegion] = useState<{
    name: string;
    sentiments: { name: string; percentage: number }[];
  } | undefined>(undefined);

  // Complete Philippine region coordinates
  const regionCoordinates = useMemo(() => {
    return {
      // Special entry for whole country
      "Philippines": [12.8797, 121.7740],
      
      // Metro Manila and surrounding provinces
      "Metro Manila": [14.5995, 120.9842],
      "Batangas": [13.7565, 121.0583],
      "Rizal": [14.6042, 121.3035],
      "Laguna": [14.2691, 121.4113],
      "Bulacan": [14.7969, 120.8787],
      "Cavite": [14.4791, 120.8970],
      "Pampanga": [15.0794, 120.6200],
      
      // Luzon
      "Luzon": [16.0, 121.0],
      "Ilocos Norte": [18.1647, 120.7116],
      "Ilocos Sur": [17.5755, 120.3869],
      "La Union": [16.6159, 120.3209],
      "Pangasinan": [15.8949, 120.2863],
      "Cagayan": [17.6132, 121.7270],
      "Isabela": [16.9754, 121.8107],
      "Quirino": [16.4907, 121.5434],
      "Nueva Vizcaya": [16.3301, 121.1710],
      "Batanes": [20.4487, 121.9702],
      "Apayao": [17.9811, 121.1333],
      "Kalinga": [17.4766, 121.3629],
      "Abra": [17.5951, 120.7983],
      "Mountain Province": [17.0417, 121.1087],
      "Ifugao": [16.8303, 121.1710],
      "Benguet": [16.4023, 120.5960],
      "Tarlac": [15.4755, 120.5960],
      "Zambales": [15.5082, 120.0691],
      "Bataan": [14.6417, 120.4818],
      "Nueva Ecija": [15.5784, 121.0687],
      "Aurora": [15.9784, 121.6323],
      "Quezon": [14.0313, 122.1106],
      "Camarines Norte": [14.1389, 122.7632],
      "Camarines Sur": [13.6252, 123.1829],
      "Albay": [13.1775, 123.5280],
      "Sorsogon": [12.9433, 124.0067],
      "Catanduanes": [13.7089, 124.2422],
      "Masbate": [12.3686, 123.6417],
      "Marinduque": [13.4771, 121.9032],
      "Occidental Mindoro": [13.1024, 120.7651],
      "Oriental Mindoro": [13.0565, 121.4069],
      "Romblon": [12.5778, 122.2695],
      "Palawan": [9.8349, 118.7384],
      
      // Visayas
      "Visayas": [11.0, 124.0],
      "Cebu": [10.3157, 123.8854],
      "Bohol": [9.8500, 124.1435],
      "Negros Oriental": [9.6168, 123.0113],
      "Negros Occidental": [10.6713, 123.0036],
      "Iloilo": [10.7202, 122.5621],
      "Capiz": [11.3889, 122.6277],
      "Aklan": [11.8166, 122.0942],
      "Antique": [11.3683, 122.0645],
      "Guimaras": [10.5982, 122.6277],
      "Leyte": [10.8731, 124.8811],
      "Southern Leyte": [10.3365, 125.1717],
      "Biliran": [11.5836, 124.4651],
      "Samar": [12.0083, 125.0373],
      "Eastern Samar": [11.6508, 125.4082],
      "Northern Samar": [12.4700, 124.6451],
      "Siquijor": [9.1985, 123.5950],
      "Tacloban": [11.2543, 125.0000], // City but commonly mentioned
      
      // Mindanao
      "Mindanao": [7.5, 125.0],
      "Davao": [7.0707, 125.6087],
      "Davao del Sur": [6.7656, 125.3284],
      "Davao del Norte": [7.5619, 125.6549],
      "Davao Oriental": [7.3172, 126.5420],
      "Davao Occidental": [6.1055, 125.6083],
      "Davao de Oro": [7.3172, 126.1748], 
      "Zamboanga del Norte": [8.1527, 123.2577],
      "Zamboanga del Sur": [7.8383, 123.2968],
      "Zamboanga Sibugay": [7.5222, 122.8198],
      "Misamis Occidental": [8.3375, 123.7071],
      "Misamis Oriental": [8.5046, 124.6220],
      "Bukidnon": [8.0515, 125.0985],
      "Lanao del Norte": [8.0730, 124.2873],
      "Lanao del Sur": [7.8232, 124.4357],
      "North Cotabato": [7.1436, 124.8511],
      "South Cotabato": [6.2969, 124.8511],
      "Sultan Kudarat": [6.5069, 124.4169],
      "Sarangani": [5.9630, 125.1990],
      "Agusan del Norte": [8.9456, 125.5319],
      "Agusan del Sur": [8.1661, 126.0152],
      "Surigao del Norte": [9.7177, 125.5950],
      "Surigao del Sur": [8.7512, 126.1378],
      "Dinagat Islands": [10.1280, 125.6083],
      "Maguindanao": [6.9423, 124.4169],
      "Basilan": [6.4221, 121.9690],
      "Sulu": [6.0474, 121.0024],
      "Tawi-Tawi": [5.1339, 119.9357],
      "Cagayan de Oro": [8.4542, 124.6319], // City but commonly mentioned
      "General Santos": [6.1164, 125.1716]  // City but commonly mentioned
    };
  }, []);

  // Process data for regions and map location mentions
  const locationData = useMemo(() => {
    const data = new Map<string, {
      count: number;
      sentiments: Map<string, number>;
      coordinates: [number, number];
    }>();
    
    // Process posts to populate the map
    sentimentPosts.forEach(post => {
      if (!post.location) return;
      
      // Convert possible raw location mentions to standardized regions
      let location = post.location;
      
      // Handle main island groups if mentioned
      if (location.toLowerCase().includes('luzon')) location = 'Luzon';
      if (location.toLowerCase().includes('visayas')) location = 'Visayas';
      if (location.toLowerCase().includes('mindanao')) location = 'Mindanao';
      
      // Handle entire Philippines as special case
      if (
        location.toLowerCase().includes('philippines') || 
        location.toLowerCase().includes('pilipinas') ||
        location.toLowerCase().includes('pinas')
      ) {
        location = 'Philippines'; 
      }
      
      // If coordinates not found, use center of Philippines
      const coordinates = regionCoordinates[location as keyof typeof regionCoordinates] || regionCoordinates["Philippines"];
      
      if (!data.has(location)) {
        data.set(location, {
          count: 0,
          sentiments: new Map(),
          coordinates
        });
      }
      
      const locationData = data.get(location)!;
      locationData.count++;
      
      const currentSentimentCount = locationData.sentiments.get(post.sentiment) || 0;
      locationData.sentiments.set(post.sentiment, currentSentimentCount + 1);
    });
    
    return data;
  }, [sentimentPosts, regionCoordinates]);

  // Convert location data to regions for map
  const regions = useMemo(() => {
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
      const maxPosts = Math.max(...Array.from(locationData.values()).map(d => d.count), 1);
      const intensity = (data.count / maxPosts) * 100;

      return {
        name,
        coordinates: data.coordinates,
        sentiment: dominantSentiment,
        intensity
      };
    });
  }, [locationData]);

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

  const handleRegionSelect = (region: {name: string}) => {
    const regionData = locationData.get(region.name);
    if (!regionData) return;

    const totalSentiments = Array.from(regionData.sentiments.entries()).reduce(
      (sum, [_, count]) => sum + count, 
      0
    );

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