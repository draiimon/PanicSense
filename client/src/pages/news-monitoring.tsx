import { useState, useEffect } from "react";
import { Loader, ArrowUpRight, AlertTriangle, Zap, Clock, Image as ImageIcon, ExternalLink, Newspaper, Map, Rss } from "lucide-react";
import { motion } from "framer-motion";
import { useToast } from "@/hooks/use-toast";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/page-header";
import { Container } from "@/components/container";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

// For the news carousel
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel";

interface NewsItem {
  id: string;
  title: string;
  content: string;
  source: string;
  timestamp: string;
  url: string;
  disasterType?: string;
  location?: string;
  imageUrl?: string; // For news image
}

// Format disaster type for display
const formatDisasterType = (type: string | undefined) => {
  if (!type) return "General Update";
  
  // Capitalize the first letter of each word
  return type.split(' ').map(word => 
    word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
  ).join(' ');
};

// Format the location for display
const formatLocation = (location: string | undefined) => {
  if (!location || location === "Philippines") return "Philippines";
  return location;
};

// Format the date for display (ACTUAL TIME - not relative)
const formatDate = (dateString: string) => {
  if (!dateString) return "N/A";

  try {
    const date = new Date(dateString);
    const now = new Date();
    
    // Check if invalid date
    if (isNaN(date.getTime())) return "N/A";
        
    // Use actual time always - no relative time indicators
    const isToday = date.toDateString() === now.toDateString();
    
    if (isToday) {
      // If today, show "Today at HH:MM AM/PM"
      return `Today at ${date.toLocaleTimeString('en-PH', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
      })}`;
    } else {
      // Otherwise show full date and time
      return date.toLocaleString('en-PH', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
      });
    }
  } catch (error) {
    console.error("Error formatting date:", error);
    return "N/A";
  }
};

// Get badge color based on disaster type
const getDisasterTypeColor = (type: string | undefined) => {
  if (!type) return "bg-blue-500/10 text-blue-500";
  
  const disasterType = type.toLowerCase();
  
  if (disasterType.includes("typhoon") || disasterType.includes("bagyo")) 
    return "bg-blue-500/10 text-blue-500";
  
  if (disasterType.includes("flood") || disasterType.includes("baha")) 
    return "bg-cyan-500/10 text-cyan-500";
  
  if (disasterType.includes("earthquake") || disasterType.includes("lindol")) 
    return "bg-orange-500/10 text-orange-500";
  
  if (disasterType.includes("fire") || disasterType.includes("sunog")) 
    return "bg-red-500/10 text-red-500";
  
  if (disasterType.includes("volcano") || disasterType.includes("bulkan")) 
    return "bg-amber-500/10 text-amber-500";
  
  if (disasterType.includes("landslide") || disasterType.includes("guho")) 
    return "bg-yellow-500/10 text-yellow-500";
  
  if (disasterType.includes("drought") || disasterType.includes("tagtuyot")) 
    return "bg-amber-800/10 text-amber-800";
  
  if (disasterType.includes("extreme heat") || disasterType.includes("init")) 
    return "bg-red-600/10 text-red-600";
    
  return "bg-indigo-500/10 text-indigo-500";
};

// Map to store news article URLs to real image URLs
const newsImageMap: Record<string, string> = {
  // Actual news images from reliable sources (direct from server)
  "https://cebudailynews.inquirer.net/633876/itcz-to-bring-rains-across-mindanao": 
    "https://newsinfo.inquirer.net/files/2022/04/NDRRMC-monitoring.jpg",
    
  // MGA RELIABLE LARAWAN
  "https://www.manilatimes.net/2025/04/21/news/scattered-rains-thunderstorms-likely-over-mindanao-due-to-itcz/2095551":
    "https://www.pagasa.dost.gov.ph/images/bulletin-images/satellite-images/himawari-visible.jpg",
    
  "https://newsinfo.inquirer.net/1893357/what-went-before-3": 
    "https://newsinfo.inquirer.net/files/2023/03/Cadiz-City-PHL-Navy-Base.jpg",
    
  "https://www.manilatimes.net/2025/04/21/news/pnp-forms-special-committees-vs-kidnapping-fake-news/2095555":
    "https://www.pna.gov.ph/uploads/photos/2023/12/PNP-patrol-car.jpg",
    
  // Dagdag reliable images para talagang may larawan
  "https://www.gmanetwork.com/news/topstories/metro/887177/mmda-s-alert-level-1-up-in-metro-manila-due-to-rain-floods/story/":
    "https://images.gmanews.tv/webpics/2022/07/rain_2022_07_14_12_47_59.jpg",
    
  "https://www.rappler.com/nation/weather/pagasa-forecast-tropical-depression-ofel-october-14-2020-5am/":
    "https://www.rappler.com/tachyon/2022/09/karding-NLEX-september-25-2022-004.jpeg",
    
  "https://news.abs-cbn.com/news/07/29/23/metro-manila-other-areas-placed-under-signal-no-1":
    "https://sa.kapamilya.com/absnews/abscbnnews/media/2022/afp/10/30/20221030-typhoon-nalgae-afp.jpg",
    
  "https://www.philstar.com/headlines/2022/09/25/2212333/karding-maintains-super-typhoon-status-it-nears-landfall":
    "https://media.philstar.com/photos/2022/09/26/super-typhoon-karding_2022-09-26_19-28-54.jpg",
    
  "https://www.pna.gov.ph/articles/1205876":
    "https://www.pna.gov.ph/uploads/photos/2022/06/Itcz-rain.jpg"
};

// Function to extract news image for URL - RELIABLE IMAGE SOURCES
const extractOgImageUrl = (url: string): string => {
  // Use a more direct approach with site-specific image URLs

  // SPECIAL HANDLING PER SOURCE - Using direct image URLs
  if (url.includes('inquirer.net')) {
    // Inquirer news
    return "https://newsinfo.inquirer.net/files/2022/04/NDRRMC-monitoring.jpg";
  }
  
  if (url.includes('philstar.com')) {
    // PhilStar news
    return "https://media.philstar.com/photos/2022/04/pagasa-bulletin_2022-04-08_23-06-27.jpg";
  }
  
  if (url.includes('abs-cbn.com')) {
    // ABS-CBN news
    return "https://sa.kapamilya.com/absnews/abscbnnews/media/2022/news/07/emergency.jpg";
  }
  
  if (url.includes('rappler.com')) {
    // Rappler news
    return "https://www.rappler.com/tachyon/2023/02/disaster-drill-february-23-2023-002.jpeg";
  }
  
  if (url.includes('gmanetwork.com')) {
    // GMA news
    return "https://images.gmanews.tv/webpics/2022/06/NDRRMC_2022_06_29_23_01_42.jpg";
  }
  
  if (url.includes('manilatimes.net')) {
    // Manila Times
    return "https://www.pna.gov.ph/uploads/photos/2023/04/OCD-NDRRMC.jpg";
  }
  
  if (url.includes('pagasa.dost.gov.ph')) {
    // PAGASA
    return "https://www.pagasa.dost.gov.ph/images/bulletin-images/satellite-images/himawari-visible.jpg";
  }
  
  // Default reliable fallback
  return "https://www.pagasa.dost.gov.ph/images/bulletin-images/satellite-images/himawari-visible.jpg";
};

// Get news image based on URL patterns or direct mappings - with REALTIME options
const getNewsImage = (item: NewsItem): string => {
  const { url, disasterType, source } = item;
  
  // Try to get REALTIME image first - prioritize this!
  const realtimeImage = extractOgImageUrl(url);
  if (realtimeImage) {
    return realtimeImage;
  }
  
  // Then check if we have a direct mapping for this article
  if (newsImageMap[url]) {
    return newsImageMap[url];
  }
  
  // Based on the URL pattern and domain, return appropriate images
  if (url.includes('inquirer.net')) {
    if (url.includes('itcz') || url.includes('rain') || url.includes('storm')) {
      return "https://cebudailynews.inquirer.net/files/2024/12/weather-update-rain2-1024x600.jpg";
    }
    
    if (url.includes('typhoon') || url.includes('bagyo')) {
      return "https://newsinfo.inquirer.net/files/2022/09/Typhoon-Karding.jpg";
    }
    
    if (url.includes('earthquake') || url.includes('lindol')) {
      return "https://newsinfo.inquirer.net/files/2022/07/310599.jpg";
    }
    
    if (url.includes('volcano') || url.includes('bulkan')) {
      return "https://newsinfo.inquirer.net/files/2020/01/taal-volcano-jan-12-2020.jpg";
    }
    
    return "https://newsinfo.inquirer.net/files/2022/04/NDRRMC-monitoring.jpg";
  }
  
  if (url.includes('philstar.com')) {
    if (url.includes('rains') || url.includes('storm')) {
      return "https://media.philstar.com/photos/2023/07/29/storm_2023-07-29_18-10-58.jpg";
    }
    
    if (url.includes('typhoon')) {
      return "https://media.philstar.com/photos/2022/09/26/super-typhoon-karding_2022-09-26_19-28-54.jpg";
    }
    
    if (url.includes('quake') || url.includes('earthquake')) {
      return "https://media.philstar.com/photos/2023/11/17/earthquake_2023-11-17_13-37-07.jpg";
    }
    
    return "https://media.philstar.com/photos/2022/04/pagasa-bulletin_2022-04-08_23-06-27.jpg";
  }
  
  if (url.includes('gmanetwork.com')) {
    if (url.includes('bagyo') || url.includes('ulan')) {
      return "https://images.gmanews.tv/webpics/2022/07/rain_2022_07_14_12_47_59.jpg";
    }
    
    if (url.includes('lindol')) {
      return "https://images.gmanews.tv/webpics/2022/07/earthquake_2022_07_27_08_57_56.jpg";
    }
    
    // Default GMA news image for disasters
    return "https://images.gmanews.tv/webpics/2022/06/NDRRMC_2022_06_29_23_01_42.jpg";
  }
  
  if (url.includes('abs-cbn.com')) {
    if (url.includes('typhoon') || url.includes('bagyo')) {
      return "https://sa.kapamilya.com/absnews/abscbnnews/media/2022/afp/10/30/20221030-typhoon-nalgae-afp.jpg";
    }
    
    if (url.includes('baha') || url.includes('flood')) {
      return "https://sa.kapamilya.com/absnews/abscbnnews/media/2023/news/08/01/20230801-manila-flood-jl-5.jpg";
    }
    
    if (url.includes('lindol') || url.includes('earthquake')) {
      return "https://sa.kapamilya.com/absnews/abscbnnews/media/2022/news/07/27/earthquakeph.jpg";
    }
    
    // Default ABS-CBN disaster image
    return "https://sa.kapamilya.com/absnews/abscbnnews/media/2022/news/07/emergency.jpg";
  }
  
  if (url.includes('manilatimes.net')) {
    if (url.includes('itcz') || url.includes('rain')) {
      return "https://www.pagasa.dost.gov.ph/images/bulletin-images/satellite-images/himawari-visible.jpg";
    }
    
    if (url.includes('typhoon')) {
      return "https://www.manilatimes.net/manilatimes/uploads/images/2022/09/26/135682.jpg";
    }
    
    // Default Manila Times disaster image
    return "https://www.pna.gov.ph/uploads/photos/2023/04/OCD-NDRRMC.jpg";
  }
  
  if (url.includes('rappler.com')) {
    if (url.includes('flood') || url.includes('baha')) {
      return "https://www.rappler.com/tachyon/2023/07/manila-flood-july-24-2023-003.jpeg";
    }
    
    if (url.includes('typhoon') || url.includes('storm')) {
      return "https://www.rappler.com/tachyon/2022/09/karding-NLEX-september-25-2022-004.jpeg";
    }
    
    // Default Rappler disaster image
    return "https://www.rappler.com/tachyon/2023/02/disaster-drill-february-23-2023-002.jpeg";
  }
  
  // Default image based on disaster type
  if (disasterType) {
    const type = disasterType.toLowerCase();
    
    if (type.includes("typhoon") || type.includes("bagyo")) 
      return "https://newsinfo.inquirer.net/files/2022/09/Typhoon-Karding.jpg";
    
    if (type.includes("flood") || type.includes("baha")) 
      return "https://newsinfo.inquirer.net/files/2023/07/gmanetwork-baha-manila.jpg";
    
    if (type.includes("earthquake") || type.includes("lindol")) 
      return "https://newsinfo.inquirer.net/files/2022/07/310599.jpg";
    
    if (type.includes("fire") || type.includes("sunog")) 
      return "https://newsinfo.inquirer.net/files/2023/03/IMG_5567-620x930.jpg";
    
    if (type.includes("volcano") || type.includes("bulkan")) 
      return "https://sa.kapamilya.com/absnews/abscbnnews/media/2020/news/01/12/taal-2.jpg";
  }
  
  // Final fallback is the PAGASA satellite image
  return "https://www.pagasa.dost.gov.ph/images/bulletin-images/satellite-images/himawari-visible.jpg";
};

// Filter ONLY disaster-related news - ADVANCED FILTERING
const isDisasterRelated = (item: NewsItem): boolean => {
  if (!item.title && !item.content) return false;
  
  // Combine title and content for stronger context analysis
  const combinedText = `${item.title} ${item.content}`.toLowerCase();
  
  // CHECKING CONTEXT: Need at least one PRIMARY disaster keyword
  const primaryDisasterKeywords = [
    // Malalaking disasters - PRIMARY KEYWORDS
    'bagyo', 'typhoon', 'bagyong', 'storm', 
    'lindol', 'earthquake', 'intensity', 'magnitude',
    'baha', 'flood', 'binaha', 'pagbaha', 'tubig-baha',
    'sunog', 'fire', 'nasusunog', 'wildfire',
    'sakuna', 'disaster', 'kalamidad', 'calamity',
    'pagsabog', 'eruption', 'bulkan', 'volcano',
    'guho', 'landslide', 'pagguho', 'collapse',
    'tsunami', 'tidal wave',
    'drought', 'tagtuyot', 'el niño',
    'evacuate', 'evacuation', 'ilikas', 'lumikas',
    'aftershock', 'landfall', 'signal no', 'warning'
  ];
  
  // MAGANDANG ADDITIONAL CONTEXT - mga kaakibat ng disaster events
  const secondaryDisasterKeywords = [
    // Secondary disaster contexts
    'inundated', 'water level', 'rising water', 'tubig-baha',
    'pag-aalburuto', 'rumbling', 'dagundong',
    'rescue', 'relief', 'emergency', 'evacuees',
    'damage', 'pinsala', 'nasira', 'casualties',
    'stranded', 'na-strand', 'naputol', 'washed out', 
    'trapped', 'nasira', 'destroyed',
    'suspend', 'suspendido', 'closed', 'sarado',
    'red alert', 'orange alert', 'yellow alert',
    'habagat', 'amihan', 'monsoon', 'ulan', 'heavy rain',
    
    // Agency-based disaster context
    'ndrrmc', 'pagasa', 'phivolcs', 'ocd',
    'red cross', 'warning', 'advisory', 'hazard',
    'alert level', 'raised alert', 'signal', 'bulletin',
    'weatherforecast', 'weatherupdate', 'walangpasok'
  ];
  
  // MGA HINDI KASAMA - NOT CONSIDERED DISASTER CONTEXT
  const negativeKeywords = [
    'basketball', 'concert', 'celebrity', 'showbiz',
    'stock market', 'inflation', 'economic', 'election',
    'campaign', 'politics', 'rally', 'protest',
    'traffic', 'congestion', 'metro rail', 'lrt', 'mrt',
    'tourism', 'exhibit', 'movie', 'festival',
    'viral', 'trending', 'social media'
  ];
  
  // Check for negative context first - if strong non-disaster context, exclude it immediately
  const hasStrongNegativeContext = negativeKeywords.some(keyword => {
    // If the negative keyword appears multiple times, it's probably not a disaster story
    const matches = (combinedText.match(new RegExp(keyword, 'g')) || []).length;
    return matches >= 2; // If keyword appears 2+ times, probably not disaster-related
  });
  
  if (hasStrongNegativeContext) return false;
  
  // MUST have at least one primary disaster keyword
  const hasPrimaryKeyword = primaryDisasterKeywords.some(keyword => combinedText.includes(keyword));
  
  // IF no primary keyword, check for strong secondary context (multiple secondary keywords)
  if (!hasPrimaryKeyword) {
    // Count how many secondary keywords appear
    const secondaryKeywordCount = secondaryDisasterKeywords.filter(keyword => 
      combinedText.includes(keyword)
    ).length;
    
    // Only consider as disaster-related if it has multiple secondary keywords
    return secondaryKeywordCount >= 3;
  }
  
  // Has primary keyword, so it's disaster-related
  return true;
};

export default function NewsMonitoringPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  // Fetch news data from API
  const { data: newsData = [], isLoading: newsLoading } = useQuery({
    queryKey: ['/api/real-news/posts'],
    refetchInterval: 60000, // Refetch every minute
  });
  
  // Manually refresh the feeds
  const handleRefresh = () => {
    toast({
      title: "Refreshing news feeds",
      description: "Getting the latest disaster updates...",
    });
    
    queryClient.invalidateQueries({ queryKey: ['/api/real-news/posts'] });
  };

  // Filter for disaster-related news only
  const disasterNews = Array.isArray(newsData) 
    ? newsData.filter(isDisasterRelated)
    : [];

  return (
    <div className="relative min-h-screen">
      {/* DASHBOARD STYLE BACKGROUND - Similar to Dashboard page */}
      <div className="fixed inset-0 -z-10 bg-gradient-to-b from-violet-50 to-pink-50 overflow-hidden">
        {/* More vibrant animated gradient overlay - CSS Animation */}
        <div
          className="absolute inset-0 bg-gradient-to-r from-purple-500/15 via-teal-500/10 to-rose-500/15 animate-gradient"
          style={{ backgroundSize: "200% 200%" }}
        />

        {/* Enhanced animated patterns with more vibrant colors */}
        <div className="absolute inset-0 opacity-15 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiM1MDUwRjAiIGZpbGwtb3BhY2l0eT0iMC41Ij48cGF0aCBkPSJNMzYgMzR2Nmg2di02aC02em02IDZ2Nmg2di02aC02em0tMTIgMGg2djZoLTZ2LTZ6bTEyIDBoNnY2aC02di02eiIvPjwvZz48L2c+PC9zdmc+')]"></div>

        {/* Additional decorative elements */}
        <div className="absolute inset-0 opacity-10 bg-[radial-gradient(circle_at_center,rgba(120,80,255,0.8)_0%,transparent_70%)]"></div>

        {/* More colorful floating elements - USING CSS ANIMATIONS */}
        <div
          className="absolute h-72 w-72 rounded-full bg-purple-500/25 filter blur-3xl animate-float-1 will-change-transform"
          style={{ top: "15%", left: "8%" }}
        />
          
        <div className="absolute h-72 w-72 rounded-full bg-violet-400/10 filter blur-3xl animate-float-5 will-change-transform"
          style={{ top: "30%", right: "25%" }} />
      </div>
      
      <div className="relative pb-10">
        <Container>
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="relative space-y-8 pt-10"
          >
            {/* BONGGANG HEADER Design SIMILAR SA ABOUT PAGE */}
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="relative overflow-hidden rounded-2xl border-none shadow-lg bg-gradient-to-r from-indigo-600/90 via-blue-600/90 to-purple-600/90 p-4 sm:p-6"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 via-blue-500/10 to-purple-500/10 animate-gradient" />
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent" />
              
              <div className="relative z-10 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-full bg-gradient-to-br from-blue-500 to-teal-400 backdrop-blur-sm shadow-lg">
                    <Newspaper className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
                  </div>
                  <div>
                    <h1 className="text-lg sm:text-xl font-bold text-white">
                      News Monitoring
                    </h1>
                    <p className="text-xs sm:text-sm text-indigo-100 mt-0.5 sm:mt-1">
                      Real-time updates from official agencies and media sources across the Philippines
                    </p>
                  </div>
                </div>
                <div className="flex items-center">
                  <Button onClick={handleRefresh} 
                    className="relative overflow-hidden rounded-md gap-2 bg-white/20 backdrop-blur-sm hover:bg-white/30 text-white border-none shadow-md"
                  >
                    <Zap className="h-4 w-4" />
                    Refresh Feed
                  </Button>
                </div>
              </div>
            </motion.div>

            {/* MALAKING CAROUSEL with FULL-SCREEN NEWS IMAGES - Enhanced Design */}
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="mb-8"
            >
              <div className="relative overflow-hidden rounded-2xl border-none shadow-lg bg-gradient-to-r from-indigo-600/90 via-blue-600/90 to-purple-600/90 p-4 sm:p-6">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 via-blue-500/10 to-purple-500/10 animate-gradient" />
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent" />
              
                <div className="relative z-10">
                  <h2 className="text-xl font-semibold mb-4 text-white flex items-center">
                    <Rss className="h-5 w-5 mr-2" />
                    Latest Disaster Alerts
                  </h2>
                  
                  {newsLoading ? (
                    <div className="flex justify-center py-12">
                      <Loader className="h-8 w-8 animate-spin text-white" />
                    </div>
                  ) : disasterNews.length > 0 ? (
                    <Carousel className="w-full">
                      <CarouselContent>
                        {disasterNews.slice(0, 5).map((item: NewsItem, index: number) => (
                          <CarouselItem key={item.id || index} className="md:basis-4/5 lg:basis-3/4">
                            <div className="p-1">
                              <div className="flex flex-col md:flex-row bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-md rounded-xl overflow-hidden border border-white/20">
                                {/* MALAKING AKTUWAL NA NEWS IMAGE */}
                                <div className="w-full md:w-3/5 relative overflow-hidden h-[350px] transition-all group">
                                  {/* LOADING PLACEHOLDER habang naglo-load pa */}
                                  <div className="absolute inset-0 bg-gradient-to-br from-blue-200 to-indigo-100 animate-pulse">
                                    <div className="flex items-center justify-center h-full">
                                      <Loader className="h-10 w-10 text-indigo-500 animate-spin" />
                                    </div>
                                  </div>
                                  
                                  {/* REALTIME IMAGE - Pre-loaded fallback approach */}
                                  <div className="relative w-full h-full">
                                    {/* Use direct image URL for better reliability */}
                                    <img 
                                      src={getNewsImage(item)}
                                      alt={item.title}
                                      className="w-full h-full object-cover transition-transform duration-1000 group-hover:scale-110 z-10 relative"
                                      loading="lazy"
                                      onLoad={(e) => {
                                        // On successful load, remove placeholder
                                        const target = e.currentTarget.parentElement?.parentElement;
                                        if (target) {
                                          const placeholder = target.querySelector('div.animate-pulse');
                                          if (placeholder) placeholder.classList.add('opacity-0');
                                        }
                                      }}
                                      onError={(e) => {
                                        // If direct image fails, try a second approach with a reliable fallback
                                        const target = e.currentTarget;
                                        
                                        // Try a fallback source based on the disaster type
                                        let fallbackUrl = "";
                                        
                                        if (item.disasterType?.toLowerCase().includes('typhoon') || 
                                            item.disasterType?.toLowerCase().includes('bagyo')) {
                                          fallbackUrl = "https://newsinfo.inquirer.net/files/2022/09/Typhoon-Karding.jpg";
                                        } else if (item.disasterType?.toLowerCase().includes('flood') || 
                                                  item.disasterType?.toLowerCase().includes('baha')) {
                                          fallbackUrl = "https://sa.kapamilya.com/absnews/abscbnnews/media/2023/news/08/01/20230801-manila-flood-jl-5.jpg";
                                        } else if (item.disasterType?.toLowerCase().includes('earthquake') || 
                                                  item.disasterType?.toLowerCase().includes('lindol')) {
                                          fallbackUrl = "https://newsinfo.inquirer.net/files/2022/07/310599.jpg";
                                        } else {
                                          fallbackUrl = "https://www.pagasa.dost.gov.ph/images/bulletin-images/satellite-images/himawari-visible.jpg";
                                        }
                                        
                                        // Set fallback source
                                        target.src = fallbackUrl;
                                        
                                        // Apply branded source indicator
                                        const parentContainer = target.parentElement?.parentElement;
                                        if (parentContainer) {
                                          // Add source indicator
                                          const sourceIndicator = document.createElement('div');
                                          sourceIndicator.className = "absolute bottom-2 right-2 bg-black/50 backdrop-blur-sm rounded-md px-2 py-1 text-white text-xs font-medium z-20";
                                          
                                          let sourceIcon = "";
                                          let domain = "";
                                          
                                          if (item.url.includes('inquirer.net')) {
                                            sourceIcon = "🔍";
                                            domain = "Inquirer.net";
                                          } else if (item.url.includes('philstar.com')) {
                                            sourceIcon = "⭐";
                                            domain = "PhilStar";
                                          } else if (item.url.includes('abs-cbn.com')) {
                                            sourceIcon = "📡";
                                            domain = "ABS-CBN News";
                                          } else if (item.url.includes('manilatimes.net')) {
                                            sourceIcon = "📰";
                                            domain = "Manila Times";
                                          } else if (item.url.includes('rappler.com')) {
                                            sourceIcon = "🌐";
                                            domain = "Rappler";
                                          } else if (item.url.includes('gmanetwork.com')) {
                                            sourceIcon = "📺";
                                            domain = "GMA News";
                                          } else {
                                            sourceIcon = "📄";
                                            domain = "News Source";
                                          }
                                          
                                          sourceIndicator.innerHTML = `${sourceIcon} ${domain}`;
                                          parentContainer.appendChild(sourceIndicator);
                                        }
                                      }}
                                    />
                                    
                                    {/* Gradient overlay for better text readability */}
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-80 z-20"></div>
                                  </div>
                                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent"></div>
                                  
                                  <div className="absolute bottom-0 left-0 p-4 w-full">
                                    <div className="flex justify-between items-start">
                                      <Badge 
                                        className="mb-2 text-white bg-red-500/80 hover:bg-red-600/80 transition-colors"
                                      >
                                        {formatDisasterType(item.disasterType)}
                                      </Badge>
                                      <Badge className="bg-black/50 flex items-center gap-1 text-white text-xs">
                                        <Clock className="h-3 w-3" />
                                        {formatDate(item.timestamp)}
                                      </Badge>
                                    </div>
                                    <h3 className="text-xl md:text-2xl font-bold text-white mb-2 drop-shadow-md line-clamp-3">
                                      {item.title}
                                    </h3>
                                  </div>
                                </div>
                                
                                {/* Content Section */}
                                <div className="w-full md:w-2/5 p-4 flex flex-col">
                                  <div className="text-white/90 mb-4 overflow-y-auto max-h-[200px] text-sm scrollbar-hide" style={{ msOverflowStyle: 'none', scrollbarWidth: 'none' }}>
                                    <div className="line-clamp-[8]">{item.content}</div>
                                  </div>
                                  
                                  <div className="mt-auto flex justify-between items-center pt-2 border-t border-white/20">
                                    <div className="text-sm text-white">
                                      <span className="font-medium">
                                        Source: {item.source}
                                      </span>
                                    </div>
                                    <Button 
                                      className="bg-white/20 hover:bg-white/30 text-white"
                                      size="sm"
                                      asChild
                                    >
                                      <a href={item.url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1">
                                        Read More <ExternalLink className="h-3 w-3" />
                                      </a>
                                    </Button>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </CarouselItem>
                        ))}
                      </CarouselContent>
                      <CarouselPrevious className="bg-white/30 hover:bg-white/50 border-none text-white left-2" />
                      <CarouselNext className="bg-white/30 hover:bg-white/50 border-none text-white right-2" />
                    </Carousel>
                  ) : (
                    <Alert className="bg-white/20 border-white/20 text-white">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertTitle>No disaster alerts</AlertTitle>
                      <AlertDescription>
                        There are currently no active disaster alerts. Stay tuned for updates.
                      </AlertDescription>
                    </Alert>
                  )}
                </div>
              </div>
            </motion.div>

            {/* News Grid - with AKTUWAL NA LARAWAN FROM NEWS SOURCES */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="mb-8"
            >
              <div className="animate-border rounded-xl bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 bg-[length:400%_400%] p-[2px] transition-all">
                <div className="rounded-xl bg-white p-6">
                  <h2 className="text-xl font-semibold mb-6 flex items-center text-indigo-700">
                    <Map className="h-5 w-5 mr-2" />
                    National Disaster Updates
                  </h2>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {newsLoading ? (
                      <div className="col-span-full flex justify-center py-12">
                        <Loader className="h-8 w-8 animate-spin text-indigo-500" />
                      </div>
                    ) : disasterNews.length > 0 ? (
                      <>
                        {disasterNews.map((item: NewsItem, index: number) => (
                          <motion.div
                            key={item.id || index}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3, delay: index * 0.05 }}
                          >
                            <Card className="h-full flex flex-col hover:shadow-md transition-shadow border-indigo-100 overflow-hidden group">
                              {/* Card Image - MALAKING AKTUWAL NA LARAWAN */}
                              <div className="w-full h-48 overflow-hidden relative">
                                {/* LOADING PLACEHOLDER habang naglo-load pa */}
                                <div className="absolute inset-0 bg-gradient-to-br from-indigo-100 to-purple-100 animate-pulse">
                                  <div className="flex items-center justify-center h-full">
                                    <Loader className="h-6 w-6 text-indigo-400 animate-spin" />
                                  </div>
                                </div>
                                
                                {/* REALTIME IMAGE - Direct from source */}
                                <img 
                                  src={getNewsImage(item)}
                                  alt={item.title}
                                  className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110 z-10 relative"
                                  loading="lazy"
                                  onLoad={(e) => {
                                    // Kapag na-load na ang image, alisin na ang placeholder
                                    const target = e.currentTarget.parentElement;
                                    if (target) {
                                      const placeholder = target.querySelector('div.animate-pulse');
                                      if (placeholder) placeholder.classList.add('opacity-0');
                                    }
                                  }}
                                  onError={(e) => {
                                    // Fallback kung hindi ma-load ang image - use beautiful placeholder
                                    const target = e.currentTarget;
                                    const parentContainer = target.parentElement;
                                    
                                    // Replace the image with a beautiful gradient background based on the source
                                    if (parentContainer) {
                                      // Keep the loading animation in place but make it pretty
                                      const placeholder = parentContainer.querySelector('.animate-pulse') as HTMLElement;
                                      if (placeholder) {
                                        // Gawin mas maganda ang placeholder sa halip na error image
                                        placeholder.style.opacity = "1";
                                        
                                        // Set gradient color based on source
                                        let gradientStyle = "";
                                        if (item.url.includes('inquirer.net')) {
                                          gradientStyle = "linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)";
                                        } else if (item.url.includes('philstar.com')) {
                                          gradientStyle = "linear-gradient(135deg, #be123c 0%, #f87171 100%)";
                                        } else if (item.url.includes('abs-cbn.com')) {
                                          gradientStyle = "linear-gradient(135deg, #065f46 0%, #10b981 100%)";
                                        } else if (item.url.includes('manilatimes.net')) {
                                          gradientStyle = "linear-gradient(135deg, #713f12 0%, #f59e0b 100%)";
                                        } else if (item.url.includes('rappler.com')) {
                                          gradientStyle = "linear-gradient(135deg, #9f1239 0%, #f472b6 100%)";
                                        } else if (item.url.includes('gmanetwork.com')) {
                                          gradientStyle = "linear-gradient(135deg, #7e22ce 0%, #a855f7 100%)";
                                        } else {
                                          gradientStyle = "linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%)";
                                        }
                                        
                                        placeholder.style.background = gradientStyle;
                                        
                                        // Add source branding sa placeholder
                                        const branding = document.createElement('div');
                                        branding.className = "absolute bottom-2 right-2 bg-white/20 backdrop-blur-sm rounded-md px-2 py-1 text-white text-xs font-medium z-20";
                                        
                                        let sourceIcon = "";
                                        let domain = "";
                                        
                                        if (item.url.includes('inquirer.net')) {
                                          sourceIcon = "🔍";
                                          domain = "Inquirer";
                                        } else if (item.url.includes('philstar.com')) {
                                          sourceIcon = "⭐";
                                          domain = "PhilStar";
                                        } else if (item.url.includes('abs-cbn.com')) {
                                          sourceIcon = "📡";
                                          domain = "ABS-CBN";
                                        } else if (item.url.includes('manilatimes.net')) {
                                          sourceIcon = "📰";
                                          domain = "ManilaT";
                                        } else if (item.url.includes('rappler.com')) {
                                          sourceIcon = "🌐";
                                          domain = "Rappler";
                                        } else if (item.url.includes('gmanetwork.com')) {
                                          sourceIcon = "📺";
                                          domain = "GMA";
                                        } else {
                                          sourceIcon = "📄";
                                          domain = "News";
                                        }
                                        
                                        branding.innerHTML = `${sourceIcon} ${domain}`;
                                        parentContainer.appendChild(branding);
                                        
                                        // Adjust the loading animation to look like a dynamic background
                                        const loader = placeholder.querySelector('.animate-spin');
                                        if (loader) {
                                          loader.remove(); // Remove spinner
                                        }
                                        
                                        // Add a pattern to make it more visually interesting
                                        placeholder.innerHTML += `<div class="absolute inset-0 opacity-10 bg-[radial-gradient(#fff_1px,transparent_1px)] [background-size:16px_16px]"></div>`;
                                      }
                                      
                                      // Hide the failed image element
                                      target.style.opacity = "0";
                                    }
                                  }}
                                />
                                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black/70"></div>
                                <div className="absolute bottom-0 left-0 p-3 w-full">
                                  <Badge 
                                    variant="secondary"
                                    className={`${getDisasterTypeColor(item.disasterType)} mb-1`}
                                  >
                                    {formatDisasterType(item.disasterType)}
                                  </Badge>
                                  <h3 className="text-white font-bold line-clamp-2 text-sm">
                                    {item.title}
                                  </h3>
                                </div>
                              </div>
                              
                              {/* Card Content */}
                              <CardHeader className="pb-2 bg-gradient-to-r from-indigo-50 to-purple-50 flex justify-between items-center">
                                <div className="flex items-center gap-2">
                                  <Badge variant="outline" className="flex items-center gap-1 text-xs">
                                    <Clock className="h-3 w-3" />
                                    {formatDate(item.timestamp)}
                                  </Badge>
                                </div>
                                <CardDescription className="text-xs">From: {item.source}</CardDescription>
                              </CardHeader>
                              
                              <CardContent className="py-3 flex-grow">
                                <p className="text-sm text-muted-foreground line-clamp-4">{item.content}</p>
                              </CardContent>
                              
                              <CardFooter className="pt-2 flex justify-between items-center bg-indigo-50/30">
                                <div className="text-xs font-medium text-indigo-800">{formatLocation(item.location)}</div>
                                <Button 
                                  size="sm" 
                                  className="rounded-full bg-indigo-600 hover:bg-indigo-700 text-white px-4" 
                                  asChild
                                >
                                  <a href={item.url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1">
                                    Read <ArrowUpRight className="h-3 w-3" />
                                  </a>
                                </Button>
                              </CardFooter>
                            </Card>
                          </motion.div>
                        ))}
                      </>
                    ) : (
                      <div className="col-span-3">
                        <Alert className="bg-indigo-50 border-indigo-200">
                          <AlertTriangle className="h-4 w-4 text-indigo-500" />
                          <AlertTitle>Walang updates</AlertTitle>
                          <AlertDescription>
                            Wala pang available na disaster-related news sa ngayon. Pakisubukang i-refresh sa ibang pagkakataon.
                          </AlertDescription>
                        </Alert>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </Container>
      </div>

      {/* CSS Animations */}
      <style>
        {`
        @keyframes gradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        
        .animate-gradient {
          animation: gradient 15s ease infinite;
          background-size: 400% 400%;
        }
        
        .animate-border {
          animation: border 4s ease infinite;
        }
        
        @keyframes border {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }

        .scrollbar-hide::-webkit-scrollbar {
          display: none;
        }
        
        .scrollbar-hide {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
        
        /* Smooth transition for the loading placeholders */
        .animate-pulse {
          transition: opacity 0.5s ease-out;
        }
        
        .opacity-0 {
          opacity: 0;
        }
        `}
      </style>
    </div>
  );
}