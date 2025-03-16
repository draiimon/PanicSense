import { useState, useMemo, useEffect } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { SentimentMap } from "@/components/analysis/sentiment-map";
import { SentimentLegend } from "@/components/analysis/sentiment-legend";
import { motion, AnimatePresence } from "framer-motion";
import { Globe, MapPin, Map, AlertTriangle, Satellite, Eye, EyeOff, BarChart3 } from "lucide-react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { getCoordinates, extractLocations } from "@/lib/geocoding";
import { toast } from "@/hooks/use-toast";

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
  const [detectedLocations, setDetectedLocations] = useState<Record<string, [number, number]>>({});

  // Complete Philippine region coordinates
  const regionCoordinates: Record<string, [number, number]> = {
    // Default coordinates for unknown locations
    "Unknown": [12.8797, 121.7740],

    // Metro Manila and surrounding provinces
    "Metro Manila": [14.5995, 120.9842],
    "Manila": [14.5995, 120.9842],
    "Batangas": [13.7565, 121.0583],
    "Rizal": [14.6042, 121.3035],
    "Laguna": [14.2691, 121.4113],
    "Bulacan": [14.7969, 120.8787],
    "Cavite": [14.2829, 120.8686],
    "Pampanga": [15.0794, 120.6200],

    // Main regions
    "Luzon": [16.0, 121.0],
    "Visayas": [11.0, 124.0],
    "Mindanao": [7.5, 125.0],

    // Major cities
    "Cebu": [10.3157, 123.8854],
    "Davao": [7.0707, 125.6087],
    "Quezon City": [14.6760, 121.0437],
    "Tacloban": [11.2543, 125.0000],
    "Baguio": [16.4023, 120.5960],
    "Zamboanga": [6.9214, 122.0790],
    "Cagayan de Oro": [8.4542, 124.6319],
    "General Santos": [6.1164, 125.1716],
    
    // Metro Manila cities
    "Makati": [14.5547, 121.0244],
    "Pasig": [14.5764, 121.0851],
    "Taguig": [14.5176, 121.0509],
    "Marikina": [14.6507, 121.1029],
    "Mandaluyong": [14.5794, 121.0359],
    "Pasay": [14.5378, 121.0014],
    "Parañaque": [14.4793, 121.0198],
    "Caloocan": [14.6499, 120.9809],
    "Muntinlupa": [14.4081, 121.0415],
    "San Juan": [14.6019, 121.0355],
    "Las Piñas": [14.4453, 120.9833],
    "Valenzuela": [14.7011, 120.9830],
    "Navotas": [14.6688, 120.9427],
    "Malabon": [14.6681, 120.9574],
    "Pateros": [14.5446, 121.0685],
    
    // Other major cities and locations
    "Angeles": [15.1450, 120.5887],
    "Bacolod": [10.6713, 122.9511],
    "Iloilo": [10.7202, 122.5621],
    "Monumento": [14.6543, 120.9834],
    "Cabanatuan": [15.4886, 120.9691],
    "Boracay": [11.9804, 121.9189],
    "Palawan": [9.8349, 118.7384],
    "Bohol": [9.8500, 124.1435],
    "Leyte": [11.0105, 124.6514],
    "Samar": [11.5750, 124.9749],
    "Pangasinan": [15.8949, 120.2863],
    "Tarlac": [15.4755, 120.5963],
    "Cagayan": [17.6132, 121.7270],
    "Bicol": [13.4213, 123.4136],
    "Nueva Ecija": [15.5784, 120.9716],
    "Benguet": [16.4023, 120.5960],
    "Albay": [13.1776, 123.5280],
    "Zambales": [15.5082, 120.0697]
  };

  // Function to find closest matching location from known regions
  const findClosestLocation = (input: string): string | null => {
    if (!input) return null;
    
    // Normalize the input
    const normalizedInput = input.toLowerCase().trim();
    
    // First check for exact matches
    for (const location of Object.keys(regionCoordinates).concat(Object.keys(detectedLocations))) {
      if (location.toLowerCase() === normalizedInput) {
        return location;
      }
    }
    
    // Then check for partial matches (location name contains the input or vice versa)
    for (const location of Object.keys(regionCoordinates).concat(Object.keys(detectedLocations))) {
      if (location.toLowerCase().includes(normalizedInput) || 
          normalizedInput.includes(location.toLowerCase())) {
        return location;
      }
    }
    
    // Finally check for typos with simple edit distance (if word is at least 4 chars)
    if (normalizedInput.length >= 4) {
      let bestMatch = null;
      let lowestDistance = Infinity;
      
      for (const location of Object.keys(regionCoordinates).concat(Object.keys(detectedLocations))) {
        // Simple edit distance calculation
        const distance = calculateEditDistance(normalizedInput, location.toLowerCase());
        if (distance < lowestDistance && distance <= 2) { // Allow 2 character differences max
          lowestDistance = distance;
          bestMatch = location;
        }
      }
      
      return bestMatch;
    }
    
    return null;
  };
  
  // Simple Levenshtein distance calculation for typo correction
  const calculateEditDistance = (a: string, b: string): number => {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;
  
    const matrix = [];
  
    // Initialize matrix
    for (let i = 0; i <= b.length; i++) {
      matrix[i] = [i];
    }
    for (let j = 0; j <= a.length; j++) {
      matrix[0][j] = j;
    }
  
    // Fill matrix
    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        if (b.charAt(i - 1) === a.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1, // substitution
            matrix[i][j - 1] + 1,     // insertion
            matrix[i - 1][j] + 1      // deletion
          );
        }
      }
    }
  
    return matrix[b.length][a.length];
  };

  // Effect for processing new posts and extracting locations
  useEffect(() => {
    const processNewPosts = async () => {
      const newLocations: Record<string, [number, number]> = {};

      for (const post of sentimentPosts) {
        // Extract locations from post text
        const extractedLocations = extractLocations(post.text);

        for (const location of extractedLocations) {
          // Check if we already have this location or a close match
          const existingLocation = findClosestLocation(location);
          
          if (existingLocation) {
            // We found a close match - no need to geocode
            continue;
          }
          
          if (!newLocations[location] && !detectedLocations[location]) {
            const coordinates = await getCoordinates(location);
            if (coordinates) {
              newLocations[location] = coordinates;
              toast({
                title: "New Location Detected",
                description: `Found and pinned: ${location}`,
                variant: "default",
              });
            }
          }
        }
      }

      if (Object.keys(newLocations).length > 0) {
        setDetectedLocations(prev => ({
          ...prev,
          ...newLocations
        }));
      }
    };

    processNewPosts();
  }, [sentimentPosts, detectedLocations]);

  // Process data for regions and map location mentions
  // List of Philippine locations to detect in text - expanded with more locations
  const philippineLocations = [
    'Manila', 'Quezon City', 'Cebu', 'Davao', 'Mindanao', 'Luzon',
    'Visayas', 'Palawan', 'Boracay', 'Baguio', 'Bohol', 'Iloilo',
    'Batangas', 'Zambales', 'Pampanga', 'Bicol', 'Leyte', 'Samar',
    'Pangasinan', 'Tarlac', 'Cagayan', 'Bulacan', 'Cavite', 'Laguna', 
    'Rizal', 'Nueva Ecija', 'Benguet', 'Albay', 'Marikina', 'Pasig',
    'Makati', 'Mandaluyong', 'Pasay', 'Taguig', 'Parañaque', 'Caloocan',
    'Metro Manila', 'Monumento', 'San Juan', 'Las Piñas', 'Muntinlupa',
    'Valenzuela', 'Navotas', 'Malabon', 'Tacloban', 'General Santos',
    'Cagayan de Oro', 'Zamboanga', 'Angeles', 'Bacolod', 'Cabanatuan'
  ];

  const locationData = useMemo(() => {
    const data: Record<string, LocationData> = {};

    // Helper function to normalize location names with typo correction
    const normalizeLocation = (loc: string): string => {
      if (!loc) return '';
      
      const lowerLoc = loc.toLowerCase().trim();
      
      // Common abbreviations and alternative names
      if (lowerLoc.includes('manila') && !lowerLoc.includes('metro')) return 'Manila';
      if (lowerLoc.includes('quezon') && lowerLoc.includes('city')) return 'Quezon City';
      if (lowerLoc === 'ncr') return 'Metro Manila';
      if (lowerLoc === 'mm') return 'Metro Manila';
      if (lowerLoc === 'qc') return 'Quezon City';
      if (lowerLoc === 'cdo') return 'Cagayan de Oro';
      if (lowerLoc === 'gensan') return 'General Santos';
      
      // Check for close matches and typos using our typo correction function
      const correctedLocation = findClosestLocation(loc);
      if (correctedLocation) {
        // If we found a match from our known locations, use that instead
        return correctedLocation;
      }
      
      // Default formatting - capitalize first letter of each word
      return loc.split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
    };

    // Process posts to populate the map with enhanced location detection
    for (const post of sentimentPosts) {
      // Get both explicit locations and try to detect from text
      const locations: string[] = [];
      
      // 1. Add explicit location from post.location if available
      if (post.location) {
        locations.push(normalizeLocation(post.location));
      }

      // 2. Add AI extracted locations (from post.text NLP analysis)
      extractLocations(post.text).forEach(loc => {
        const normalizedLoc = normalizeLocation(loc);
        if (!locations.includes(normalizedLoc)) {
          locations.push(normalizedLoc);
        }
      });
      
      // 3. Add locations from direct text matching (like dashboard)
      const postText = post.text.toLowerCase();
      for (const location of philippineLocations) {
        if (postText.includes(location.toLowerCase())) {
          const normalizedLoc = normalizeLocation(location);
          if (!locations.includes(normalizedLoc)) {
            locations.push(normalizedLoc);
          }
        }
      }

      // Process each detected location
      for (const location of locations) {
        // Skip generic mentions
        if (
          location.toLowerCase() === 'not specified' ||
          location.toLowerCase() === 'not mentioned' ||
          location.toLowerCase() === 'none' ||
          location.toLowerCase().includes('unspecified') ||
          location.toLowerCase().includes('not mentioned')
        ) {
          continue;
        }

        // Get coordinates from predefined list or detected locations
        let coordinates = regionCoordinates[location] ?? detectedLocations[location];

        if (!coordinates) {
          // Skip locations we can't pin
          continue;
        }

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
      }
    }

    return data;
  }, [sentimentPosts, detectedLocations]);

  // Convert location data to regions for map
  const regions = useMemo((): Region[] => {
    const processedData = new Map<string, any>();
    
    sentimentPosts.forEach(post => {
      if (post.location) {
        const fullLocation = post.location;
        // Split location if it contains comma (e.g. "Taytay, Rizal" -> "Rizal")
        const mainRegion = fullLocation.includes(',') ? 
          fullLocation.split(',')[1].trim() : fullLocation;
          
        if (!processedData.has(mainRegion)) {
          processedData.set(mainRegion, {
            count: 0,
            sentiments: {},
            disasterTypes: {},
            coordinates: regionCoordinates[mainRegion] || regionCoordinates['Unknown']
          });
        }
        
        const data = processedData.get(mainRegion);
        data.count++;
        data.sentiments[post.sentiment] = (data.sentiments[post.sentiment] || 0) + 1;
        if (post.disasterType) {
          data.disasterTypes[post.disasterType] = (data.disasterTypes[post.disasterType] || 0) + 1;
        }
      }
    });

    return Array.from(processedData.entries()).map(([name, data]) => {
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
      let maxTypeCount = 0;
      let dominantDisasterType: string | undefined;

      Object.entries(data.disasterTypes).forEach(([type, count]) => {
        if (count > maxTypeCount) {
          maxTypeCount = count;
          dominantDisasterType = type;
        }
      });

      // Calculate intensity based on post count relative to maximum
      const maxPosts = Math.max(...Object.values(locationData).map(d => d.count), 1);
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