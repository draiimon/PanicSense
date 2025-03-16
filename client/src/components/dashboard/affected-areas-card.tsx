import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { getSentimentBadgeClasses } from "@/lib/colors";
import { getDisasterTypeColor } from "@/lib/colors";
import { 
  MapPin, 
  AlertTriangle, 
  TrendingUp, 
  MapPinned, 
  Flame, 
  Droplets, 
  Wind, 
  Mountain 
} from "lucide-react";
import { SentimentPost } from "@/lib/api";
import { useIsMobile } from "@/hooks/use-mobile";

interface AffectedAreaProps {
  sentimentPosts: SentimentPost[];
  isLoading?: boolean;
}

interface AffectedArea {
  name: string;
  sentiment: string;
  disasterType: string | null;
  impactLevel: number;
}

// Get disaster type icon based on type
function getDisasterIcon(type: string | null) {
  if (!type) return <MapPin className="h-4 w-4 text-gray-500" />;
  
  switch (type.toLowerCase()) {
    case 'flood':
      return <Droplets className="h-4 w-4" style={{ color: getDisasterTypeColor(type) }} />;
    case 'fire':
      return <Flame className="h-4 w-4" style={{ color: getDisasterTypeColor(type) }} />;
    case 'typhoon':
      return <Wind className="h-4 w-4" style={{ color: getDisasterTypeColor(type) }} />;
    case 'earthquake':
      return <MapPinned className="h-4 w-4" style={{ color: getDisasterTypeColor(type) }} />;
    case 'volcanic eruption':
    case 'volcano':
      return <Mountain className="h-4 w-4" style={{ color: getDisasterTypeColor(type) }} />;
    case 'landslide':
      return <Mountain className="h-4 w-4" style={{ color: getDisasterTypeColor(type) }} />;
    default:
      return <MapPin className="h-4 w-4" style={{ color: getDisasterTypeColor(type) }} />;
  }
}

export function AffectedAreasCard({ sentimentPosts, isLoading = false }: AffectedAreaProps) {
  const [affectedAreas, setAffectedAreas] = useState<AffectedArea[]>([]);
  const [isSlotRolling, setIsSlotRolling] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const isMobile = useIsMobile();

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
      .slice(0, 5); // Show only 5 areas to avoid scrolling

    setAffectedAreas(sortedAreas);
  }, [sentimentPosts]);

  // Slot machine effect with hover pause
  useEffect(() => {
    if (affectedAreas.length === 0) return;
    
    const startSlotRolling = () => {
      if (!containerRef.current) return;
      
      const container = containerRef.current;
      const scrollHeight = container.scrollHeight / 3;
      const clientHeight = container.clientHeight;
      
      if (scrollHeight <= clientHeight) return;
      
      let currentPosition = 0;
      const scrollSpeed = 0.5;
      let animationId: number | null = null;
      let isPaused = false;

      const resetScroll = () => {
        if (currentPosition >= scrollHeight * 2) {
          currentPosition = scrollHeight;
          container.scrollTop = currentPosition;
        } else if (currentPosition <= 0) {
          currentPosition = scrollHeight;
          container.scrollTop = currentPosition;
        }
      };
      
      const scroll = () => {
        if (!containerRef.current || isPaused) {
          animationId = requestAnimationFrame(scroll);
          return;
        }
        
        currentPosition += scrollSpeed;
        
        resetScroll();
        containerRef.current.scrollTop = currentPosition;
        animationId = requestAnimationFrame(scroll);
      };
      
      // Add hover and interaction handlers
      const handleMouseEnter = () => {
        isPaused = true;
      };
      
      const handleMouseLeave = () => {
        isPaused = false;
        currentPosition = container.scrollTop;
      };
      
      const handleScroll = () => {
        if (!isPaused) {
          currentPosition = container.scrollTop;
        }
      };
      
      container.addEventListener('mouseenter', handleMouseEnter);
      container.addEventListener('mouseleave', handleMouseLeave);
      container.addEventListener('scroll', handleScroll);
      
      animationId = requestAnimationFrame(scroll);
      
      return () => {
        if (animationId) {
          cancelAnimationFrame(animationId);
        }
        container.removeEventListener('mouseenter', handleMouseEnter);
        container.removeEventListener('mouseleave', handleMouseLeave);
        container.removeEventListener('scroll', handleScroll);
      };
    };
    
    const cleanup = startSlotRolling();
    return cleanup;
  }, [affectedAreas]);

  return (
    <div 
      ref={containerRef}
      className="h-full overflow-auto scrollbar-hide relative"
      style={{ maskImage: 'linear-gradient(to bottom, transparent, black 10%, black 90%, transparent 100%)' }}
    >
      <AnimatePresence>
        <div className="space-y-4 p-4">
          {[...affectedAreas, ...affectedAreas, ...affectedAreas].map((area, index) => (
            <div className="flex flex-col items-center justify-center h-[350px] py-8">
              <div className="w-16 h-16 rounded-full bg-blue-50 flex items-center justify-center mb-4">
                <MapPin className="h-7 w-7 text-blue-400" />
              </div>
              <p className="text-center text-base text-slate-500 mb-2">No affected areas detected</p>
              <p className="text-center text-sm text-slate-400">Upload data to see disaster impact by location</p>
            </div>
          ) : (
            // Add duplicated items for infinite scrolling effect
            [...affectedAreas, ...affectedAreas, ...affectedAreas].map((area, index) => (
              <motion.div
                key={`${area.name}-${index}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ 
                  opacity: 1, 
                  y: 0, 
                  transition: { 
                    delay: index * 0.05,
                    duration: 0.4 
                  } 
                }}
                className="rounded-xl p-4 bg-white border border-blue-50 shadow-sm hover:shadow-md hover:border-blue-100 transition-all"
              >
                <div className="flex items-start justify-between">
                  <div className="flex gap-3">
                    <div className="p-2 rounded-lg bg-blue-50 flex-shrink-0">
                      {getDisasterIcon(area.disasterType)}
                    </div>
                    <div>
                      <div className="flex items-center">
                        <h3 className="font-semibold text-gray-900 text-base">{area.name}</h3>
                        <div className="flex items-center ml-2 gap-0.5">
                          <TrendingUp className="h-3 w-3 text-amber-500" />
                          <span className="text-xs font-medium text-amber-600">
                            {area.impactLevel}
                          </span>
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-1.5 mt-2">
                        <Badge 
                          className={`${getSentimentBadgeClasses(area.sentiment)} px-2.5 py-0.5 rounded-full text-xs font-medium`}
                        >
                          {area.sentiment}
                        </Badge>

                        {area.disasterType && (
                          <Badge
                            className="px-2.5 py-0.5 rounded-full text-xs font-medium"
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
                </div>
                
                {/* Impact meter */}
                <div className="mt-3">
                  <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(100, (area.impactLevel / 10) * 100)}%` }}
                      transition={{ duration: 0.5, delay: 0.2 + (index * 0.05) }}
                      className="h-full rounded-full"
                      style={{ 
                        backgroundColor: area.disasterType ? 
                          getDisasterTypeColor(area.disasterType) : 
                          getSentimentBadgeClasses(area.sentiment).includes('red') ? 
                            '#ef4444' : '#3b82f6'
                      }}
                    />
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </div>
      </AnimatePresence>
    </div>
  );
}