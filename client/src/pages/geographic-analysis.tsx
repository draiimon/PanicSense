import { useState, useMemo } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { SentimentMap } from "@/components/analysis/sentiment-map";
import { SentimentLegend } from "@/components/analysis/sentiment-legend";
import { motion, AnimatePresence } from "framer-motion";
import { Globe, MapPin, Map, AlertTriangle, RefreshCw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

// Define types for the component
interface Region {
  name: string;
  coordinates: [number, number];
  sentiment: string;
  disasterType?: string;
  intensity: number;
}

// Type for region coordinates  
type RegionCoordinates = Record<string, [number, number]>;

export default function GeographicAnalysis() {
  const [activeMapType, setActiveMapType] = useState<'disaster' | 'emotion'>('disaster');
  const { sentimentPosts, refreshData } = useDisasterContext();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [mapView, setMapView] = useState<'standard' | 'satellite'>('standard'); // Added map view state

  // Complete Philippine region coordinates
  const regionCoordinates = useMemo(() => {
    return {
      // Special entry for whole country
      "Philippines": [12.8797, 121.7740],

      // Metro Manila and surrounding provinces
      "Metro Manila": [14.5995, 120.9842],
      "Manila": [14.5995, 120.9842],
      "Batangas": [13.7565, 121.0583],
      "Rizal": [14.6042, 121.3035],
      "Laguna": [14.2691, 121.4113],
      "Bulacan": [14.7969, 120.8787],
      "Cavite": [14.2829, 120.8686],
      "Pampanga": [15.0794, 120.6200],
      "Bacoor": [14.4628, 120.8967],
      "Imus": [14.4297, 120.9367],
      "Dasmariñas": [14.3294, 120.9367],
      "General Trias": [14.3833, 120.8833],
      "Kawit": [14.4351, 120.9019],
      "Tanza": [14.3953, 120.8508],

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
      "Tacloban": [11.2543, 125.0000],

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
      "Cagayan de Oro": [8.4542, 124.6319],
      "General Santos": [6.1164, 125.1716]
    };
  }, []);

  // Process data for regions and map location mentions
  const locationData = useMemo(() => {
    const data: Record<string, {
      count: number;
      sentiments: Record<string, number>;
      disasterTypes: Record<string, number>;
      coordinates: [number, number];
    }> = {};

    // Process posts to populate the map
    sentimentPosts.forEach(post => {
      if (!post.location) return;

      // Convert possible raw location mentions to standardized regions
      let location = post.location;
      const lowerLocation = location.toLowerCase().trim();

      // Skip "not specified" and generic Philippines mentions
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
      const rawCoordinates = regionCoordinates[location as keyof typeof regionCoordinates];
      if (!rawCoordinates) return;

      // Ensure it's a tuple of exactly 2 numbers
      const coordinates: [number, number] = [rawCoordinates[0], rawCoordinates[1]];

      // Create location data if it doesn't exist yet
      if (!data[location]) {
        data[location] = {
          count: 0,
          sentiments: {},
          disasterTypes: {},
          coordinates
        };
      }

      // Update counts
      data[location].count++;

      // Track sentiments
      const currentSentimentCount = data[location].sentiments[post.sentiment] || 0;
      data[location].sentiments[post.sentiment] = currentSentimentCount + 1;

      // Track disaster types
      if (post.disasterType) {
        const currentDisasterTypeCount = data[location].disasterTypes[post.disasterType] || 0;
        data[location].disasterTypes[post.disasterType] = currentDisasterTypeCount + 1;
      }
    });

    return data;
  }, [sentimentPosts, regionCoordinates]);

  // Convert location data to regions for map
  const regions = useMemo((): Region[] => {
    return Object.entries(locationData).map(([name, data]) => {
      // Find dominant sentiment
      let maxCount = 0;
      let dominantSentiment = "Neutral";

      // Process sentiments
      Object.entries(data.sentiments).forEach(([sentiment, count]) => {
        if (count > maxCount) {
          maxCount = count;
          dominantSentiment = sentiment;
        }
      });

      // Find dominant disaster type
      let maxDisasterCount = 0;
      let dominantDisasterType = "";

      // Process disaster types
      Object.entries(data.disasterTypes).forEach(([disasterType, count]) => {
        if (count > maxDisasterCount) {
          maxDisasterCount = count;
          dominantDisasterType = disasterType;
        }
      });

      // Calculate intensity based on post count relative to maximum
      const allCounts = Object.values(locationData).map(d => d.count);
      const maxPosts = allCounts.length > 0 ? Math.max(...allCounts) : 1;
      const intensity = (data.count / maxPosts) * 100;

      return {
        name,
        coordinates: data.coordinates,
        sentiment: dominantSentiment,
        disasterType: dominantDisasterType || undefined,
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
        disasterType: region.disasterType
      }));
  }, [regions]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refreshData();
    setIsRefreshing(false);
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card className="bg-white shadow-md border-none">
        <CardHeader className="pb-4">
          <CardTitle className="text-2xl font-bold flex items-center gap-2">
            <Globe className="h-6 w-6 text-blue-600" />
            Geographic Analysis
          </CardTitle>
          <p className="text-sm text-slate-500">
            Visualizing disaster impact and emotional response distribution across Philippine regions
          </p>
        </CardHeader>
      </Card>

      <div className="bg-white rounded-lg shadow-md">
        {/* Controls Header */}
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          {/* Map Type Tabs */}
          <div className="flex space-x-2">
            <button
              className={`px-6 py-2 font-medium text-sm rounded-lg transition-all flex items-center gap-2 ${
                activeMapType === 'disaster'
                  ? 'bg-blue-50 text-blue-600 border border-blue-200'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
              }`}
              onClick={() => setActiveMapType('disaster')}
            >
              <AlertTriangle className="h-4 w-4" />
              Disaster Impact
            </button>
            <button
              className={`px-6 py-2 font-medium text-sm rounded-lg transition-all flex items-center gap-2 ${
                activeMapType === 'emotion'
                  ? 'bg-blue-50 text-blue-600 border border-blue-200'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
              }`}
              onClick={() => setActiveMapType('emotion')}
            >
              <MapPin className="h-4 w-4" />
              Sentiment Distribution
            </button>
          </div>

          {/* Map Controls */}
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              className={`flex items-center gap-2 ${isRefreshing ? 'opacity-50' : ''}`}
              onClick={handleRefresh}
              disabled={isRefreshing}
            >
              <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
              Refresh Data
            </Button>
            <div className="bg-slate-100 rounded-lg p-1 flex">
              <Button
                size="sm"
                variant={mapView === 'standard' ? 'default' : 'outline'}
                className="rounded-r-none px-3 h-8"
                onClick={() => setMapView('standard')}
              >
                <Map className="h-4 w-4 mr-1" />
                <span className="text-xs">Standard</span>
              </Button>
              <Button
                size="sm"
                variant={mapView === 'satellite' ? 'default' : 'outline'}
                className="rounded-l-none px-3 h-8"
                onClick={() => setMapView('satellite')}
              >
                <Layers className="h-4 w-4 mr-1" />
                <span className="text-xs">Satellite</span>
              </Button>
            </div>
          </div>
        </div>

        {/* Map and Legend Container */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-6">
          {/* Map Container */}
          <div className="lg:col-span-2">
            <Card className="border-none shadow-sm">
              <CardContent className="p-4">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeMapType}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <SentimentMap
                      regions={regions}
                      colorBy={activeMapType === 'disaster' ? 'disasterType' : 'sentiment'}
                    />
                  </motion.div>
                </AnimatePresence>
              </CardContent>
            </Card>
          </div>

          {/* Legend Container */}
          <div>
            <SentimentLegend
              mostAffectedAreas={mostAffectedAreas}
              showRegionSelection={false}
              colorBy={activeMapType === 'disaster' ? 'disasterType' : 'sentiment'}
            />
          </div>
        </div>
      </div>
    </div>
  );
}