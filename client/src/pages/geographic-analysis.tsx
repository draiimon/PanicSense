import { useState, useMemo } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { SentimentMap } from "@/components/analysis/sentiment-map";
import { SentimentLegend } from "@/components/analysis/sentiment-legend";
import { motion, AnimatePresence } from "framer-motion";
import { Globe, MapPin, Map, AlertTriangle, Satellite, Eye, EyeOff, BarChart3 } from "lucide-react";
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
  const [showMarkers, setShowMarkers] = useState<boolean>(true);

  // Complete Philippine region coordinates
  const regionCoordinates = {
    // Default coordinates for unknown locations
    "Unknown": [12.8797, 121.7740] as [number, number],

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

    // Main regions
    "Luzon": [16.0, 121.0] as [number, number],
    "Visayas": [11.0, 124.0] as [number, number],
    "Mindanao": [7.5, 125.0] as [number, number],

    // Major cities
    "Cebu": [10.3157, 123.8854] as [number, number],
    "Davao": [7.0707, 125.6087] as [number, number],
    "Quezon City": [14.6760, 121.0437] as [number, number],
    "Tacloban": [11.2543, 125.0000] as [number, number],
    "Baguio": [16.4023, 120.5960] as [number, number],
    "Zamboanga": [6.9214, 122.0790] as [number, number],
    "Cagayan de Oro": [8.4542, 124.6319] as [number, number],
    "General Santos": [6.1164, 125.1716] as [number, number],

    // Provinces
    "Ilocos Norte": [18.1647, 120.7116] as [number, number],
    "Ilocos Sur": [17.5755, 120.3869] as [number, number],
    "La Union": [16.6159, 120.3209] as [number, number],
    "Pangasinan": [15.8949, 120.2863] as [number, number],
    "Cagayan": [17.6132, 121.7270] as [number, number],
    "Isabela": [16.9754, 121.8107] as [number, number],
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
    "Mindoro": [13.1024, 120.7651] as [number, number],
    "Palawan": [9.8349, 118.7384] as [number, number],
    "Bohol": [9.8500, 124.1435] as [number, number],
    "Leyte": [10.8731, 124.8811] as [number, number],
    "Samar": [12.0083, 125.0373] as [number, number],
    "Iloilo": [10.7202, 122.5621] as [number, number],
    "Capiz": [11.3889, 122.6277] as [number, number],
    "Aklan": [11.8166, 122.0942] as [number, number],
    "Antique": [11.3683, 122.0645] as [number, number],
    "Negros Occidental": [10.6713, 123.0036] as [number, number],
    "Negros Oriental": [9.6168, 123.0113] as [number, number],
    "Zamboanga del Norte": [8.1527, 123.2577] as [number, number],
    "Zamboanga del Sur": [7.8383, 123.2968] as [number, number],
    "Lanao del Norte": [8.0730, 124.2873] as [number, number],
    "Lanao del Sur": [7.8232, 124.4357] as [number, number],
    "Bukidnon": [8.0515, 125.0985] as [number, number],
    "Davao del Sur": [6.7656, 125.3284] as [number, number],
    "Davao del Norte": [7.5619, 125.6549] as [number, number],
    "Davao Oriental": [7.3172, 126.5420] as [number, number],
    "South Cotabato": [6.2969, 124.8511] as [number, number],
    "North Cotabato": [7.1436, 124.8511] as [number, number],
    "Sultan Kudarat": [6.5069, 124.4169] as [number, number],
    "Maguindanao": [6.9423, 124.4169] as [number, number],
    "Agusan del Norte": [8.9456, 125.5319] as [number, number],
    "Agusan del Sur": [8.1661, 126.0152] as [number, number],
    "Surigao del Norte": [9.7177, 125.5950] as [number, number],
    "Surigao del Sur": [8.7512, 126.1378] as [number, number],
  };

  // Process data for regions and map location mentions
  const locationData = useMemo(() => {
    const data: Record<string, LocationData> = {};

    // Helper function to normalize location names
    const normalizeLocation = (loc: string): string => {
      const lowerLoc = loc.toLowerCase().trim();
      // Handle different variations of location names
      if (lowerLoc.includes('manila') && !lowerLoc.includes('metro')) return 'Manila';
      if (lowerLoc.includes('quezon') && lowerLoc.includes('city')) return 'Quezon City';
      if (lowerLoc === 'ncr') return 'Metro Manila';
      if (lowerLoc === 'mm') return 'Metro Manila';
      if (lowerLoc === 'qc') return 'Quezon City';
      if (lowerLoc === 'cdo') return 'Cagayan de Oro';
      if (lowerLoc === 'gensan') return 'General Santos';
      if (lowerLoc === 'mindoro occidental') return 'Mindoro';
      if (lowerLoc === 'mindoro oriental') return 'Mindoro';

      // Return capitalized version of the location
      return loc.split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
    };

    // Process posts to populate the map
    sentimentPosts.forEach(post => {
      if (!post.location) return;

      let location = normalizeLocation(post.location);

      // Skip generic mentions
      if (
        location.toLowerCase() === 'not specified' ||
        location.toLowerCase() === 'not mentioned' ||
        location.toLowerCase() === 'none'
      ) {
        return;
      }

      // Get coordinates or use default if not found
      const coordinates = regionCoordinates[location as keyof typeof regionCoordinates] || regionCoordinates["Unknown"];

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
      data[location].sentiments[post.sentiment] = (data[location].sentiments[post.sentiment] || 0) + 1;

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
    <div className="flex flex-col min-h-screen bg-slate-50">
      {/* Content Container - Mobile-optimized layout */}
      <div className="flex-1 px-2 sm:px-4 py-2 sm:py-4 overflow-y-auto">
        {/* Header Card - More compact on mobile */}
        <Card className="bg-white shadow-sm border-none mb-2 sm:mb-4">
          <CardHeader className="p-2 sm:p-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <Globe className="h-5 w-5 sm:h-6 sm:w-6 text-blue-600" />
                <div>
                  <CardTitle className="text-lg sm:text-xl font-bold text-slate-800">
                    Geographic Analysis
                  </CardTitle>
                  <p className="text-xs sm:text-sm text-slate-500 mt-0.5 sm:mt-1">
                    Visualizing disaster impact across Philippine regions
                  </p>
                </div>
              </div>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setShowMarkers(!showMarkers)}
                className="flex items-center gap-1 z-10 text-xs sm:text-sm py-1 px-2 h-8"
              >
                {showMarkers ? <EyeOff className="h-3 w-3 sm:h-4 sm:w-4" /> : <Eye className="h-3 w-3 sm:h-4 sm:w-4" />}
                <span>{showMarkers ? 'Hide' : 'Show'}</span>
              </Button>
            </div>
          </CardHeader>
        </Card>

        {/* Main Content Area - Optimized for mobile */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-2 sm:gap-4">
          {/* Map Container - Better height management */}
          <div className="lg:col-span-2 bg-white shadow-sm rounded-lg overflow-hidden flex flex-col h-[calc(100vh-10rem)] sm:h-[calc(100vh-12rem)]">
            {/* Map Controls - Compact on mobile */}
            <div className="border-b border-slate-200 p-2 sm:p-4">
              <div className="flex flex-wrap gap-2 items-center justify-between">
                {/* View Type Controls */}
                <div className="flex gap-1 sm:gap-2">
                  <Button
                    variant={activeMapType === 'disaster' ? 'default' : 'outline'}
                    onClick={() => setActiveMapType('disaster')}
                    className="flex items-center gap-1 text-xs sm:text-sm h-8"
                    size="sm"
                  >
                    <AlertTriangle className="h-3 w-3 sm:h-4 sm:w-4" />
                    <span>Disaster</span>
                  </Button>
                  <Button
                    variant={activeMapType === 'emotion' ? 'default' : 'outline'}
                    onClick={() => setActiveMapType('emotion')}
                    className="flex items-center gap-1 text-xs sm:text-sm h-8"
                    size="sm"
                  >
                    <BarChart3 className="h-3 w-3 sm:h-4 sm:w-4" />
                    <span>Sentiment</span>
                  </Button>
                </div>

                {/* Map Style Controls */}
                <div className="flex rounded-lg overflow-hidden border border-slate-200">
                  <Button
                    size="sm"
                    variant={mapView === 'standard' ? 'default' : 'outline'}
                    onClick={() => setMapView('standard')}
                    className="rounded-none border-0 h-8 px-2"
                  >
                    <Map className="h-3 w-3 sm:h-4 sm:w-4" />
                  </Button>
                  <Button
                    size="sm"
                    variant={mapView === 'satellite' ? 'default' : 'outline'}
                    onClick={() => setMapView('satellite')}
                    className="rounded-none border-0 h-8 px-2"
                  >
                    <Satellite className="h-3 w-3 sm:h-4 sm:w-4" />
                  </Button>
                </div>
              </div>

              {/* Active Filters Display - Compact on mobile */}
              {selectedRegionFilter && (
                <div className="mt-2 flex items-center gap-1 sm:gap-2">
                  <span className="text-xs sm:text-sm text-slate-500">Filtered by:</span>
                  <Badge variant="secondary" className="flex items-center gap-1 text-xs">
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
                    showMarkers={showMarkers}
                  />
                </motion.div>
              </AnimatePresence>
            </div>
          </div>

          {/* Legend Panel - Adjusted for mobile */}
          <div className="lg:col-span-1 bg-white shadow-sm rounded-lg flex flex-col min-h-[300px] sm:min-h-[400px] lg:h-[calc(100vh-12rem)]">
            <div className="p-2 border-b border-slate-200">
              <h3 className="font-semibold text-slate-800 text-xs sm:text-sm">Analysis Legend</h3>
            </div>
            <div className="flex-1 overflow-y-auto">
              <SentimentLegend
                mostAffectedAreas={mostAffectedAreas}
                showRegionSelection={false}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}