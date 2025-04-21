import { useState, useEffect } from "react";
import { 
  Loader, 
  ArrowUpRight, 
  AlertTriangle, 
  Zap, 
  Clock, 
  Image as ImageIcon, 
  ExternalLink, 
  Newspaper, 
  Map, 
  Rss,
  Cloud,
  Droplets,
  Flame,
  Mountain,
  LifeBuoy,
  Thermometer
} from "lucide-react";
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

// Get news image based on the 3-tier fallback system
// TIER 1: Original image from feed
// TIER 2: Source-specific branded image
// TIER 3: Generic fallback image
const getNewsImage = (item: NewsItem): string => {
  const { url, disasterType, imageUrl } = item;
  
  // TIER 1: Use the actual image from the feed if available (highest priority)
  if (imageUrl && imageUrl.startsWith('http')) {
    return imageUrl;
  }
  
  // TIER 2: Check if we have a direct mapping for this specific URL in our collection
  if (newsImageMap[url]) {
    return newsImageMap[url];
  }
  
  // TIER 3: Use source-specific fallback images based on domain
  try {
    const urlObj = new URL(url);
    const domain = urlObj.hostname;
    
    // Return reliable source-specific images that we know will always work
    if (domain.includes('inquirer.net')) {
      return "https://newsinfo.inquirer.net/files/2022/04/NDRRMC-monitoring.jpg";
    }
    
    if (domain.includes('philstar.com')) {
      return "https://media.philstar.com/photos/2022/09/26/super-typhoon-karding_2022-09-26_19-28-54.jpg";
    }
    
    if (domain.includes('abs-cbn.com')) {
      return "https://sa.kapamilya.com/absnews/abscbnnews/media/2022/news/07/emergency.jpg";
    }
    
    if (domain.includes('rappler.com')) {
      return "https://www.rappler.com/tachyon/2022/09/karding-NLEX-september-25-2022-004.jpeg";
    }
    
    if (domain.includes('gmanetwork.com')) {
      return "https://images.gmanews.tv/webpics/2022/07/rain_2022_07_14_12_47_59.jpg";
    }
    
    if (domain.includes('manilatimes.net')) {
      return "https://www.manilatimes.net/manilatimes/uploads/images/2022/07/26/143483.jpg";
    }
    
    if (domain.includes('mindanaotimes.com')) {
      return "https://mindanaotimes.com.ph/wp-content/uploads/2023/07/RAIN.jpg";
    }
    
    if (domain.includes('bworldonline')) {
      return "https://www.bworldonline.com/wp-content/uploads/2022/09/BSP-1-300x169.jpg";
    }
    
    if (domain.includes('sunstar')) {
      return "https://www.sunstar.com.ph/uploads/images/2022/04/13/346407.jpg";
    }
    
    if (domain.includes('panaynews.net')) {
      return "https://www.panaynews.net/wp-content/uploads/2023/08/RAIN-IN-FLOOD.jpg";
    }
    
    // Final fallback for any other news source
    return "https://www.pagasa.dost.gov.ph/images/bulletin-images/satellite-images/himawari-visible.jpg";
  } catch (error) {
    // If URL parsing fails, use a generic fallback
    return "https://images.gmanews.tv/webpics/2022/06/NDRRMC_2022_06_29_23_01_42.jpg";
  }
};

// Filter ONLY disaster-related news
const isDisasterRelated = (item: NewsItem): boolean => {
  if (!item.title && !item.content) return false;
  
  // Combine title and content for stronger context analysis
  const combinedText = `${item.title} ${item.content}`.toLowerCase();
  
  // High priority disaster keywords
  const primaryKeywords = [
    'typhoon', 'bagyo', 'lindol', 'earthquake', 'baha', 'flood',
    'evacuation', 'evacuate', 'landslide', 'guho', 'tsunami',
    'landfall', 'storm signal', 'disaster', 'calamity', 'rescue',
    'storm surge', 'sunog', 'fire', 'volcano', 'bulkan',
    'magnitude', 'intensity', 'phivolcs', 'pagasa', 'ndrrmc',
    'emergency', 'warning', 'bulletin', 'advisory', 'alert level'
  ];
  
  // Medium priority disaster keywords
  const secondaryKeywords = [
    'destroyed', 'damaged', 'casualties', 'casualties', 'fatalities',
    'stranded', 'suspended', 'heavy rain', 'malakas na ulan',
    'weather', 'rescue', 'relief', 'affected', 'red cross',
    'destroyed', 'pinsala', 'nasira'
  ];
  
  // Check if any of the high priority keywords are in the text
  for (const keyword of primaryKeywords) {
    if (combinedText.includes(keyword)) {
      return true;
    }
  }
  
  // Check if multiple secondary keywords are in the text
  let secondaryCount = 0;
  for (const keyword of secondaryKeywords) {
    if (combinedText.includes(keyword)) {
      secondaryCount++;
    }
  }
  
  // Need at least 2 secondary keywords
  return secondaryCount >= 2;
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

  // Filter news data
  const allNews = Array.isArray(newsData) ? newsData : [];
  
  // Filter for disaster-related news only
  const disasterNews = allNews.filter(isDisasterRelated);

  return (
    <div className="relative min-h-screen">
      {/* DASHBOARD STYLE BACKGROUND */}
      <div className="fixed inset-0 -z-10 bg-gradient-to-b from-violet-50 to-pink-50 overflow-hidden">
        {/* Animated gradient overlay */}
        <div
          className="absolute inset-0 bg-gradient-to-r from-purple-500/15 via-teal-500/10 to-rose-500/15 animate-gradient"
          style={{ backgroundSize: "200% 200%" }}
        />

        {/* Enhanced animated patterns */}
        <div className="absolute inset-0 opacity-15 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiM1MDUwRjAiIGZpbGwtb3BhY2l0eT0iMC41Ij48cGF0aCBkPSJNMzYgMzR2Nmg2di02aC02em02IDZ2Nmg2di02aC02em0tMTIgMGg2djZoLTZ2LTZ6bTEyIDBoNnY2aC02di02eiIvPjwvZz48L2c+PC9zdmc+')]"></div>

        {/* Additional decorative elements */}
        <div className="absolute inset-0 opacity-10 bg-[radial-gradient(circle_at_center,rgba(120,80,255,0.8)_0%,transparent_70%)]"></div>

        {/* Floating elements */}
        <div
          className="absolute h-72 w-72 rounded-full bg-purple-500/25 filter blur-3xl animate-float-1 will-change-transform"
          style={{ top: "15%", left: "8%" }}
        />
          
        <div className="absolute h-72 w-72 rounded-full bg-violet-400/10 filter blur-3xl animate-float-5 will-change-transform"
          style={{ top: "30%", right: "25%" }} />
      </div>
      
      <div className="relative pb-10">
        <Container>
          <PageHeader
            heading="News Monitoring"
            subheading="Stay updated with the latest disaster alerts and Philippine news."
            className="mb-6"
          >
            <Button variant="outline" onClick={handleRefresh} className="ml-auto">
              <Rss className="h-4 w-4 mr-2" />
              Refresh Feeds
            </Button>
          </PageHeader>

          {/* MAIN CONTENT */}
          <div className="mt-6 space-y-6">
          
            {/* DISASTER ALERTS - Latest disaster feeds */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="mb-8"
            >
              <div className="animate-border rounded-xl bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 bg-[length:400%_400%] p-[2px] transition-all">
                <div className="rounded-xl bg-white p-6">
                  <h2 className="text-xl font-semibold mb-6 flex items-center text-indigo-700">
                    <Zap className="h-5 w-5 mr-2" />
                    Latest Disaster Alerts
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
                              {/* Card Image - With proper fallback system */}
                              <div className="w-full h-48 overflow-hidden relative">
                                {/* LOADING PLACEHOLDER while loading */}
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
                                    // When image loads, hide placeholder
                                    const target = e.currentTarget.parentElement;
                                    if (target) {
                                      const placeholder = target.querySelector('div.animate-pulse');
                                      if (placeholder) placeholder.classList.add('opacity-0');
                                    }
                                  }}
                                  onError={(e) => {
                                    // If image fails, use fallback gradient
                                    const target = e.currentTarget;
                                    const parentContainer = target.parentElement;
                                    
                                    if (parentContainer) {
                                      // Keep the loading animation in place but make it pretty
                                      const placeholder = parentContainer.querySelector('.animate-pulse') as HTMLElement;
                                      if (placeholder) {
                                        // Make placeholder visible
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
                                        
                                        // Add source branding
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
                                        
                                        // Adjust the loading animation pattern
                                        const loader = placeholder.querySelector('.animate-spin');
                                        if (loader) {
                                          loader.remove(); // Remove spinner
                                        }
                                        
                                        // Add a pattern to make it visually interesting
                                        placeholder.innerHTML += `<div class="absolute inset-0 opacity-10 bg-[radial-gradient(#fff_1px,transparent_1px)] [background-size:16px_16px]"></div>`;
                                      }
                                      
                                      // Hide the failed image element
                                      target.style.opacity = "0";
                                    }
                                  }}
                                />
                                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black/70"></div>
                                <div className="absolute bottom-0 left-0 p-3 w-full">
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
                                  
                                  <Badge variant="outline" className="flex items-center gap-1 text-xs">
                                    <Map className="h-3 w-3" />
                                    {formatLocation(item.location)}
                                  </Badge>
                                </div>
                              </CardHeader>
                              
                              <CardContent className="py-2 flex-1">
                                <p className="text-sm text-muted-foreground line-clamp-3">
                                  {item.content}
                                </p>
                              </CardContent>
                              
                              <CardFooter className="pt-2 pb-4">
                                <a
                                  href={item.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-xs inline-flex items-center text-indigo-600 hover:text-indigo-700 transition-colors"
                                >
                                  Read more on {item.source}
                                  <ExternalLink className="h-3 w-3 ml-1" />
                                </a>
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

            {/* GENERAL NEWS CAROUSEL - Hindi lang disaster news */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="mb-12"
            >
              <div className="rounded-xl bg-gradient-to-r from-purple-500 via-pink-500 to-orange-500 p-[2px]">
                <div className="rounded-xl bg-white p-6">
                  <h2 className="text-xl font-semibold mb-6 flex items-center text-pink-700">
                    <Newspaper className="h-5 w-5 mr-2" />
                    Latest Philippine News
                  </h2>
                  
                  {newsLoading ? (
                    <div className="flex justify-center py-12">
                      <Loader className="h-8 w-8 animate-spin text-pink-500" />
                    </div>
                  ) : !allNews.filter(item => item.title && item.content && !isDisasterRelated(item)).length ? (
                    <Alert className="bg-pink-50 border-pink-200">
                      <AlertTriangle className="h-4 w-4 text-pink-500" />
                      <AlertTitle>No general news available</AlertTitle>
                      <AlertDescription>
                        There are currently no general news articles available. Please check back later.
                      </AlertDescription>
                    </Alert>
                  ) : (
                    <Carousel
                      className="w-full"
                      opts={{
                        align: "start",
                        loop: true,
                      }}
                    >
                      <CarouselContent>
                        {allNews
                          .filter(item => item.title && item.content && !isDisasterRelated(item))
                          .map((item: NewsItem, index: number) => (
                            <CarouselItem key={item.id || index} className="md:basis-1/2 lg:basis-1/3">
                              <Card className="h-full flex flex-col hover:shadow-md transition-shadow border-pink-100 overflow-hidden group">
                                {/* Card Image - REALTIME LARAWAN */}
                                <div className="w-full h-48 overflow-hidden relative">
                                  {/* LOADING PLACEHOLDER habang naglo-load pa */}
                                  <div className="absolute inset-0 bg-gradient-to-br from-pink-100 to-purple-100 animate-pulse">
                                    <div className="flex items-center justify-center h-full">
                                      <Loader className="h-6 w-6 text-pink-400 animate-spin" />
                                    </div>
                                  </div>
                                  
                                  {/* REALTIME IMAGE - Direct from source */}
                                  <img 
                                    src={getNewsImage(item)}
                                    alt={item.title}
                                    className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110 z-10 relative"
                                    loading="lazy"
                                    onLoad={(e) => {
                                      // When image loads, hide placeholder
                                      const target = e.currentTarget.parentElement;
                                      if (target) {
                                        const placeholder = target.querySelector('div.animate-pulse');
                                        if (placeholder) placeholder.classList.add('opacity-0');
                                      }
                                    }}
                                    onError={(e) => {
                                      // If image fails, use fallback gradient
                                      const target = e.currentTarget;
                                      const parentContainer = target.parentElement;
                                      
                                      if (parentContainer) {
                                        // Keep the loading animation in place but make it pretty
                                        const placeholder = parentContainer.querySelector('.animate-pulse') as HTMLElement;
                                        if (placeholder) {
                                          // Make placeholder visible
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
                                            gradientStyle = "linear-gradient(135deg, #d946ef 0%, #ec4899 100%)";
                                          }
                                          
                                          placeholder.style.background = gradientStyle;
                                          
                                          // Add source branding
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
                                          
                                          // Adjust loading animation
                                          const loader = placeholder.querySelector('.animate-spin');
                                          if (loader) {
                                            loader.remove(); // Remove spinner
                                          }
                                          
                                          // Add pattern
                                          placeholder.innerHTML += `<div class="absolute inset-0 opacity-10 bg-[radial-gradient(#fff_1px,transparent_1px)] [background-size:16px_16px]"></div>`;
                                        }
                                        
                                        // Hide the failed image element
                                        target.style.opacity = "0";
                                      }
                                    }}
                                  />
                                  <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black/70"></div>
                                  <div className="absolute bottom-0 left-0 p-3 w-full">
                                    <h3 className="text-white font-bold line-clamp-2 text-sm">
                                      {item.title}
                                    </h3>
                                  </div>
                                </div>
                                
                                {/* Card Content */}
                                <CardHeader className="pb-2 bg-gradient-to-r from-pink-50 to-purple-50 flex justify-between items-center">
                                  <div className="flex items-center gap-2">
                                    <Badge variant="outline" className="flex items-center gap-1 text-xs">
                                      <Clock className="h-3 w-3" />
                                      {formatDate(item.timestamp)}
                                    </Badge>
                                  </div>
                                </CardHeader>
                                
                                <CardContent className="py-2 flex-1">
                                  <p className="text-sm text-muted-foreground line-clamp-3">
                                    {item.content}
                                  </p>
                                </CardContent>
                                
                                <CardFooter className="pt-2 pb-4">
                                  <a
                                    href={item.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-xs inline-flex items-center text-pink-600 hover:text-pink-700 transition-colors"
                                  >
                                    Read more on {item.source}
                                    <ExternalLink className="h-3 w-3 ml-1" />
                                  </a>
                                </CardFooter>
                              </Card>
                            </CarouselItem>
                          ))}
                      </CarouselContent>
                      <div className="flex items-center justify-center mt-6">
                        <CarouselPrevious className="relative mr-2" />
                        <CarouselNext className="relative ml-2" />
                      </div>
                    </Carousel>
                  )}
                </div>
              </div>
            </motion.div>
          </div>
        </Container>
      </div>
    </div>
  );
}