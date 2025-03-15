import { useState, useMemo } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { SentimentMap } from "@/components/analysis/sentiment-map";
import { SentimentLegend } from "@/components/analysis/sentiment-legend";
import { motion, AnimatePresence } from "framer-motion";

// Define Region type to match SentimentMap expectations
interface Region {
  name: string;
  coordinates: [number, number];
  sentiment: string;
  disasterType?: string;
  intensity: number;
}

export default function EmotionAnalysis() {
  const [activeMapType, setActiveMapType] = useState<'disaster' | 'emotion'>('emotion');
  const { sentimentPosts } = useDisasterContext();
  const [selectedRegion, setSelectedRegion] = useState<{
    name: string;
    sentiments: { name: string; percentage: number }[];
  } | null>(null);

  // Complete Philippine region coordinates
  const regionCoordinates = useMemo(() => {
    return {
      // Special entry for whole country
      "Philippines": [12.8797, 121.7740] as [number, number],

      // Metro Manila and surrounding provinces
      "Metro Manila": [14.5995, 120.9842] as [number, number],
      "Manila": [14.5995, 120.9842] as [number, number],
      "Batangas": [13.7565, 121.0583] as [number, number],
      "Rizal": [14.6042, 121.3035] as [number, number],
      "Laguna": [14.2691, 121.4113] as [number, number],
      "Bulacan": [14.7969, 120.8787] as [number, number],
      "Cavite": [14.2829, 120.8686] as [number, number],
      "Pampanga": [15.0794, 120.6200] as [number, number],
      "Bacoor": [14.4628, 120.8967] as [number, number],
      "Imus": [14.4297, 120.9367] as [number, number],
      "Dasmariñas": [14.3294, 120.9367] as [number, number],
      "General Trias": [14.3833, 120.8833] as [number, number],
      "Kawit": [14.4351, 120.9019] as [number, number],
      "Tanza": [14.3953, 120.8508] as [number, number],

      // Luzon
      "Luzon": [16.0, 121.0] as [number, number],
      "Ilocos Norte": [18.1647, 120.7116] as [number, number],
      "Ilocos Sur": [17.5755, 120.3869] as [number, number],
      "La Union": [16.6159, 120.3209] as [number, number],
      "Pangasinan": [15.8949, 120.2863] as [number, number],
      "Cagayan": [17.6132, 121.7270] as [number, number],
      "Isabela": [16.9754, 121.8107] as [number, number],
      "Quirino": [16.4907, 121.5434] as [number, number],
      "Nueva Vizcaya": [16.3301, 121.1710] as [number, number],
      "Batanes": [20.4487, 121.9702] as [number, number],
      "Apayao": [17.9811, 121.1333] as [number, number],
      "Kalinga": [17.4766, 121.3629] as [number, number],
      "Abra": [17.5951, 120.7983] as [number, number],
      "Mountain Province": [17.0417, 121.1087] as [number, number],
      "Ifugao": [16.8303, 121.1710] as [number, number],
      "Benguet": [16.4023, 120.5960] as [number, number],
      "Tarlac": [15.4755, 120.5960] as [number, number],
      "Zambales": [15.5082, 120.0691] as [number, number],
      "Bataan": [14.6417, 120.4818] as [number, number],
      "Nueva Ecija": [15.5784, 121.0687] as [number, number],
      "Aurora": [15.9784, 121.6323] as [number, number],
      "Quezon": [14.0313, 122.1106] as [number, number],
      "Camarines Norte": [14.1389, 122.7632] as [number, number],
      "Camarines Sur": [13.6252, 123.1829] as [number, number],
      "Albay": [13.1775, 123.5280] as [number, number],
      "Sorsogon": [12.9433, 124.0067] as [number, number],
      "Catanduanes": [13.7089, 124.2422] as [number, number],
      "Masbate": [12.3686, 123.6417] as [number, number],
      "Marinduque": [13.4771, 121.9032] as [number, number],
      "Occidental Mindoro": [13.1024, 120.7651] as [number, number],
      "Oriental Mindoro": [13.0565, 121.4069] as [number, number],
      "Romblon": [12.5778, 122.2695] as [number, number],
      "Palawan": [9.8349, 118.7384] as [number, number],

      // Visayas
      "Visayas": [11.0, 124.0] as [number, number],
      "Cebu": [10.3157, 123.8854] as [number, number],
      "Bohol": [9.8500, 124.1435] as [number, number],
      "Negros Oriental": [9.6168, 123.0113] as [number, number],
      "Negros Occidental": [10.6713, 123.0036] as [number, number],
      "Iloilo": [10.7202, 122.5621] as [number, number],
      "Capiz": [11.3889, 122.6277] as [number, number],
      "Aklan": [11.8166, 122.0942] as [number, number],
      "Antique": [11.3683, 122.0645] as [number, number],
      "Guimaras": [10.5982, 122.6277] as [number, number],
      "Leyte": [10.8731, 124.8811] as [number, number],
      "Southern Leyte": [10.3365, 125.1717] as [number, number],
      "Biliran": [11.5836, 124.4651] as [number, number],
      "Samar": [12.0083, 125.0373] as [number, number],
      "Eastern Samar": [11.6508, 125.4082] as [number, number],
      "Northern Samar": [12.4700, 124.6451] as [number, number],
      "Siquijor": [9.1985, 123.5950] as [number, number],
      "Tacloban": [11.2543, 125.0000] as [number, number],

      // Mindanao
      "Mindanao": [7.5, 125.0] as [number, number],
      "Davao": [7.0707, 125.6087] as [number, number],
      "Davao del Sur": [6.7656, 125.3284] as [number, number],
      "Davao del Norte": [7.5619, 125.6549] as [number, number],
      "Davao Oriental": [7.3172, 126.5420] as [number, number],
      "Davao Occidental": [6.1055, 125.6083] as [number, number],
      "Davao de Oro": [7.3172, 126.1748] as [number, number],
      "Zamboanga del Norte": [8.1527, 123.2577] as [number, number],
      "Zamboanga del Sur": [7.8383, 123.2968] as [number, number],
      "Zamboanga Sibugay": [7.5222, 122.8198] as [number, number],
      "Misamis Occidental": [8.3375, 123.7071] as [number, number],
      "Misamis Oriental": [8.5046, 124.6220] as [number, number],
      "Bukidnon": [8.0515, 125.0985] as [number, number],
      "Lanao del Norte": [8.0730, 124.2873] as [number, number],
      "Lanao del Sur": [7.8232, 124.4357] as [number, number],
      "North Cotabato": [7.1436, 124.8511] as [number, number],
      "South Cotabato": [6.2969, 124.8511] as [number, number],
      "Sultan Kudarat": [6.5069, 124.4169] as [number, number],
      "Sarangani": [5.9630, 125.1990] as [number, number],
      "Agusan del Norte": [8.9456, 125.5319] as [number, number],
      "Agusan del Sur": [8.1661, 126.0152] as [number, number],
      "Surigao del Norte": [9.7177, 125.5950] as [number, number],
      "Surigao del Sur": [8.7512, 126.1378] as [number, number],
      "Dinagat Islands": [10.1280, 125.6083] as [number, number],
      "Maguindanao": [6.9423, 124.4169] as [number, number],
      "Basilan": [6.4221, 121.9690] as [number, number],
      "Sulu": [6.0474, 121.0024] as [number, number],
      "Tawi-Tawi": [5.1339, 119.9357] as [number, number],
      "Cagayan de Oro": [8.4542, 124.6319] as [number, number],
      "General Santos": [6.1164, 125.1716] as [number, number]
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
      const lowerLocation = location.toLowerCase().trim();

      // Skip generic locations
      if (
        lowerLocation === 'not specified' ||
        lowerLocation === 'philippines' ||
        lowerLocation === 'pilipinas' ||
        lowerLocation === 'pinas'
      ) {
        return;
      }

      // Handle Manila specifically
      if (lowerLocation.includes('manila') && !lowerLocation.includes('metro')) {
        location = 'Manila';
      }

      // Handle main island groups if mentioned
      if (lowerLocation.includes('luzon')) location = 'Luzon';
      if (lowerLocation.includes('visayas')) location = 'Visayas';
      if (lowerLocation.includes('mindanao')) location = 'Mindanao';

      // If coordinates not found, skip this location
      const coordinates = regionCoordinates[location as keyof typeof regionCoordinates];
      if (!coordinates) return;

      if (!data.has(location)) {
        data.set(location, {
          count: 0,
          sentiments: new Map(),
          coordinates
        });
      }

      const locationData = data.get(location)!;
      locationData.count++;

      // Track sentiments
      const currentSentimentCount = locationData.sentiments.get(post.sentiment) || 0;
      locationData.sentiments.set(post.sentiment, currentSentimentCount + 1);
    });

    return data;
  }, [sentimentPosts, regionCoordinates]);

  // Convert location data to regions for map
  const regions = useMemo<Region[]>(() => {
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
      const intensity = (data.count / (maxPosts || 1)) * 100;

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
      .slice(0, 5)
      .map(region => ({
        name: region.name,
        sentiment: region.sentiment,
        disasterType: undefined
      }));
  }, [regions]);

  const handleRegionSelect = (region: { name: string }) => {
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
    <div className="container mx-auto p-4 space-y-6">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-white rounded-lg shadow-md p-6"
      >
        <h1 className="text-2xl font-bold text-slate-800">Emotion Analysis</h1>
        <p className="mt-2 text-slate-600">
          Analyze emotional responses and sentiment patterns across different regions
        </p>
      </motion.div>

      {/* Map and Legend Container */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Map Container */}
        <div className="lg:col-span-2">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeMapType}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-white rounded-lg shadow-md p-4"
            >
              <SentimentMap
                regions={regions}
                onRegionSelect={handleRegionSelect}
                colorBy="sentiment"
              />
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Legend Container */}
        <div className="lg:col-span-1">
          <SentimentLegend
            mostAffectedAreas={mostAffectedAreas}
            selectedRegion={selectedRegion}
            colorBy="sentiment"
          />
        </div>
      </div>
    </div>
  );
}