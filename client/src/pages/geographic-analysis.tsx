import { useState, useMemo } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { SentimentMap } from "@/components/analysis/sentiment-map";
import { SentimentLegend } from "@/components/analysis/sentiment-legend";
import { motion, AnimatePresence } from "framer-motion";
import { Globe, MapPin, Map, AlertTriangle, RefreshCw, Satellite } from "lucide-react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// Define types for the component
interface Region {
  name: string;
  coordinates: [number, number];
  sentiment: string;
  disasterType?: string;
  intensity: number;
}

interface LocationData {
  count: number;
  sentiments: Record<string, number>;
  disasterTypes: Record<string, number>;
  coordinates: [number, number];
}

export default function GeographicAnalysis() {
  const [activeMapType, setActiveMapType] = useState<'disaster' | 'emotion'>('disaster');
  const { sentimentPosts, refreshData } = useDisasterContext();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [mapView, setMapView] = useState<'standard' | 'satellite'>('standard');
  const [selectedRegionFilter, setSelectedRegionFilter] = useState<string | null>(null);

  // Philippine region coordinates
  const regionCoordinates = {
    "Philippines": [12.8797, 121.7740] as [number, number],
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

  // Process data for regions and map location mentions
  const locationData = useMemo(() => {
    const data: Record<string, LocationData> = {};

    // Process posts to populate the map
    sentimentPosts.forEach(post => {
      if (!post.location) return;

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
      const coordinates = regionCoordinates[location as keyof typeof regionCoordinates];
      if (!coordinates) return;

      // Initialize location data if it doesn't exist
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
      data[location].sentiments[post.sentiment] = (data[location].sentiments[post.sentiment] || 0) + 1;

      // Track disaster types
      if (post.disasterType) {
        data[location].disasterTypes[post.disasterType] = (data[location].disasterTypes[post.disasterType] || 0) + 1;
      }
    });

    return data;
  }, [sentimentPosts]);

  // Convert location data to regions for map
  const regions = useMemo((): Region[] => {
    return Object.entries(locationData).map(([name, data]) => {
      // Find dominant sentiment
      let maxCount = 0;
      let dominantSentiment = "Neutral";

      Object.entries(data.sentiments).forEach(([sentiment, count]) => {
        if (count > maxCount) {
          maxCount = count;
          dominantSentiment = sentiment;
        }
      });

      // Find dominant disaster type
      let maxDisasterCount = 0;
      let dominantDisasterType: string | undefined;

      Object.entries(data.disasterTypes).forEach(([disasterType, count]) => {
        if (count > maxDisasterCount) {
          maxDisasterCount = count;
          dominantDisasterType = disasterType;
        }
      });

      // Calculate intensity based on post count relative to maximum
      const maxPosts = Math.max(...Object.values(locationData).map(d => d.count));
      const intensity = (data.count / maxPosts) * 100;

      return {
        name,
        coordinates: data.coordinates,
        sentiment: dominantSentiment,
        disasterType: dominantDisasterType,
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
    <div className="flex flex-col h-screen bg-slate-50">
      {/* Content Container - Adjust for navbar height */}
      <div className="flex-1 flex flex-col lg:flex-row gap-4 p-4 h-[calc(100vh-4rem)] overflow-hidden">
        {/* Left Panel - Map and Controls */}
        <div className="flex-1 flex flex-col gap-4 min-w-0">
          {/* Header Card */}
          <Card className="bg-white shadow-sm border-none">
            <CardHeader className="p-4">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-2">
                  <Globe className="h-6 w-6 text-blue-600" />
                  <div>
                    <CardTitle className="text-xl font-bold text-slate-800">
                      Geographic Analysis
                    </CardTitle>
                    <p className="text-sm text-slate-500 mt-1">
                      Visualizing disaster impact across Philippine regions
                    </p>
                  </div>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleRefresh}
                  disabled={isRefreshing}
                  className="flex items-center gap-2"
                >
                  <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
                  <span className="hidden sm:inline">Refresh Data</span>
                </Button>
              </div>
            </CardHeader>
          </Card>

          {/* Main Content Area */}
          <div className="flex flex-col lg:flex-row gap-4 flex-1 min-h-0">
            {/* Map Container */}
            <div className="flex-1 flex flex-col bg-white shadow-sm rounded-lg overflow-hidden">
              {/* Map Controls */}
              <div className="border-b border-slate-200 p-4">
                <div className="flex flex-wrap gap-4 items-center justify-between">
                  {/* View Type Controls */}
                  <div className="flex gap-2">
                    <Button
                      variant={activeMapType === 'disaster' ? 'default' : 'outline'}
                      onClick={() => setActiveMapType('disaster')}
                      className="flex items-center gap-2"
                      size="sm"
                    >
                      <AlertTriangle className="h-4 w-4" />
                      <span className="hidden sm:inline">Disaster</span>
                    </Button>
                    <Button
                      variant={activeMapType === 'emotion' ? 'default' : 'outline'}
                      onClick={() => setActiveMapType('emotion')}
                      className="flex items-center gap-2"
                      size="sm"
                    >
                      <MapPin className="h-4 w-4" />
                      <span className="hidden sm:inline">Emotion</span>
                    </Button>
                  </div>

                  {/* Map Style Controls */}
                  <div className="flex rounded-lg overflow-hidden border border-slate-200">
                    <Button
                      size="sm"
                      variant={mapView === 'standard' ? 'default' : 'outline'}
                      onClick={() => setMapView('standard')}
                      className="rounded-none border-0"
                    >
                      <Map className="h-4 w-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant={mapView === 'satellite' ? 'default' : 'outline'}
                      onClick={() => setMapView('satellite')}
                      className="rounded-none border-0"
                    >
                      <Satellite className="h-4 w-4" />
                    </Button>
                  </div>
                </div>

                {/* Active Filters Display */}
                {selectedRegionFilter && (
                  <div className="mt-3 flex items-center gap-2">
                    <span className="text-sm text-slate-500">Filtered by:</span>
                    <Badge variant="secondary" className="flex items-center gap-1">
                      {selectedRegionFilter}
                      <button
                        onClick={() => setSelectedRegionFilter(null)}
                        className="ml-1 hover:text-red-500"
                      >
                        ×
                      </button>
                    </Badge>
                  </div>
                )}
              </div>

              {/* Map View Container */}
              <div className="relative flex-1 min-h-0">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeMapType}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="absolute inset-0"
                  >
                    <SentimentMap
                      regions={regions}
                      mapType={activeMapType}
                      view={mapView}
                    />
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>

            {/* Legend Panel - Now always visible with mobile optimization */}
            <div className="w-full lg:w-80 h-auto lg:h-full bg-white shadow-sm rounded-lg overflow-hidden">
              <div className="p-4 border-b border-slate-200">
                <h3 className="font-semibold text-slate-800">Analysis Legend</h3>
              </div>
              <div className="overflow-y-auto h-[calc(100%-4rem)]">
                <SentimentLegend
                  mostAffectedAreas={mostAffectedAreas}
                  showRegionSelection={false}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}