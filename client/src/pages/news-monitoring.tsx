import { useState, useEffect } from "react";
import { Loader, ArrowUpRight, AlertTriangle, Zap, Clock, Image as ImageIcon, ExternalLink } from "lucide-react";
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

// Format the date for display (relative time)
const formatDate = (dateString: string) => {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  
  // Fix: For future dates or incorrect timestamps, show "Just now" instead of negative
  if (diffMs < 0) return "Just now";
  
  const diffSec = Math.round(diffMs / 1000);
  const diffMin = Math.round(diffSec / 60);
  const diffHour = Math.round(diffMin / 60);
  const diffDay = Math.round(diffHour / 24);

  if (diffSec < 60) return `${diffSec} sec ago`;
  if (diffMin < 60) return `${diffMin} min ago`;
  if (diffHour < 24) return `${diffHour} hr ago`;
  if (diffDay < 30) return `${diffDay} days ago`;
  
  // Format date nicely
  return date.toLocaleDateString('en-PH', {
    year: 'numeric',
    month: 'short', 
    day: 'numeric'
  });
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
  // Actual news images for specific articles
  "https://cebudailynews.inquirer.net/633876/itcz-to-bring-rains-across-mindanao": 
    "https://cebudailynews.inquirer.net/files/2024/12/weather-update-rain2-1024x600.jpg",
    
  // Add more direct image mappings here as you discover them
  "https://www.manilatimes.net/2025/04/21/news/scattered-rains-thunderstorms-likely-over-mindanao-due-to-itcz/2095551":
    "https://www.pagasa.dost.gov.ph/images/bulletin-images/satellite-images/himawari-visible.jpg",
    
  "https://newsinfo.inquirer.net/1893357/what-went-before-3": 
    "https://newsinfo.inquirer.net/files/2023/03/Cadiz-City-PHL-Navy-Base.jpg",
    
  "https://www.manilatimes.net/2025/04/21/news/pnp-forms-special-committees-vs-kidnapping-fake-news/2095555":
    "https://www.pna.gov.ph/uploads/photos/2023/12/PNP-patrol-car.jpg"
};

// Get news image based on URL patterns or direct mappings
const getNewsImage = (item: NewsItem): string => {
  const { url, disasterType, source } = item;
  
  // First check if we have a direct mapping for this article
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

// Filter only disaster-related news
const isDisasterRelated = (item: NewsItem): boolean => {
  if (!item.title && !item.content) return false;
  
  const combinedText = `${item.title} ${item.content}`.toLowerCase();
  
  const disasterKeywords = [
    // Tagalog terms
    'bagyo', 'lindol', 'baha', 'sunog', 'sakuna', 'kalamidad', 'pagsabog', 'bulkan',
    'pagputok', 'guho', 'tagtuyot', 'init', 'pagguho', 'habagat', 'pinsala', 'tsunami',
    'salanta', 'ulan', 'dagundong', 'likas', 'evacuate', 'evacuation',
    
    // English terms
    'typhoon', 'earthquake', 'flood', 'fire', 'disaster', 'calamity', 'eruption', 'volcano',
    'landslide', 'drought', 'heat wave', 'tsunami', 'storm', 'damage', 'tremor', 'aftershock',
    'evacuation', 'emergency', 'relief', 'rescue', 'warning', 'alert', 'ndrrmc', 'pagasa', 'phivolcs'
  ];
  
  return disasterKeywords.some(keyword => combinedText.includes(keyword));
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
      {/* Animated Background */}
      <div className="fixed inset-0 -z-10 bg-gradient-to-b from-violet-50 to-pink-50 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-500/15 via-teal-500/10 to-rose-500/15 animate-gradient"
          style={{ backgroundSize: '400% 400%', animation: 'gradient 15s ease infinite' }}
        />
        <div className="absolute inset-0 opacity-25">
          <div className="absolute inset-0 bg-[radial-gradient(#e5e7eb_1px,transparent_1px)] [background-size:20px_20px]" />
        </div>
      </div>
      
      <Container>
        <PageHeader
          heading="Real-Time Disaster News Monitoring"
          subheading="Monitor calamity and disaster-related news in real-time from official government agencies and media sources"
          className="mb-8 relative z-10"
        >
          <Button onClick={handleRefresh} 
            className="relative overflow-hidden rounded-md gap-2 bg-gradient-to-r from-indigo-600 via-blue-600 to-purple-600 hover:from-indigo-500 hover:via-blue-500 hover:to-purple-500 shadow-md"
          >
            <Zap className="h-4 w-4" />
            Refresh Feed
          </Button>
        </PageHeader>

        {/* MALAKING CAROUSEL with FULL-SCREEN NEWS IMAGES */}
        <div className="mb-8">
          <div className="relative overflow-hidden rounded-2xl border-none shadow-lg bg-gradient-to-r from-indigo-600/90 via-blue-600/90 to-purple-600/90 p-4">
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 via-blue-500/10 to-purple-500/10 animate-gradient" />
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent" />
            
            <div className="relative z-10">
              <h2 className="text-xl font-semibold mb-4 text-white flex items-center">
                <AlertTriangle className="h-5 w-5 mr-2" />
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
                              <img 
                                src={getNewsImage(item)}
                                alt={item.title}
                                className="w-full h-full object-cover transition-transform duration-1000 group-hover:scale-110"
                                onError={(e) => {
                                  // Fallback if the primary image fails to load
                                  const target = e.currentTarget;
                                  
                                  // Attempt to fix image URL if it's from a known source
                                  if (target.src.includes('inquirer.net')) {
                                    target.src = "https://newsinfo.inquirer.net/files/2022/04/NDRRMC-monitoring.jpg";
                                  } else if (target.src.includes('philstar.com')) {
                                    target.src = "https://media.philstar.com/photos/2022/04/pagasa-bulletin_2022-04-08_23-06-27.jpg";
                                  } else if (target.src.includes('abs-cbn.com')) {
                                    target.src = "https://sa.kapamilya.com/absnews/abscbnnews/media/2022/news/07/emergency.jpg";
                                  } else if (target.src.includes('manilatimes.net')) {
                                    target.src = "https://www.pna.gov.ph/uploads/photos/2023/04/OCD-NDRRMC.jpg";
                                  } else {
                                    // Final fallback
                                    target.src = "https://www.pagasa.dost.gov.ph/images/bulletin-images/satellite-images/himawari-visible.jpg";
                                  }
                                }}
                              />
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
                              <div className="text-white/90 mb-4 overflow-y-auto max-h-[200px] text-sm">
                                {item.content}
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
        </div>

        {/* News Grid - with AKTUWAL NA LARAWAN FROM NEWS SOURCES */}
        <div className="mb-8 animate-border rounded-xl bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 bg-[length:400%_400%] p-[2px] transition-all">
          <div className="rounded-xl bg-white p-6">
            <h2 className="text-xl font-semibold mb-6 flex items-center text-indigo-700">
              <AlertTriangle className="h-5 w-5 mr-2" />
              Disaster News Feed
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
                          <img 
                            src={getNewsImage(item)} 
                            alt={item.title}
                            className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                            onError={(e) => {
                              // Fallback if the primary image fails to load
                              const target = e.currentTarget;
                              
                              // Check source to use appropriate fallback
                              if (target.src.includes('inquirer.net')) {
                                target.src = "https://newsinfo.inquirer.net/files/2022/04/NDRRMC-monitoring.jpg";
                              } else if (target.src.includes('philstar.com')) {
                                target.src = "https://media.philstar.com/photos/2022/04/pagasa-bulletin_2022-04-08_23-06-27.jpg";
                              } else if (target.src.includes('abs-cbn.com')) {
                                target.src = "https://sa.kapamilya.com/absnews/abscbnnews/media/2022/news/07/emergency.jpg";
                              } else if (target.src.includes('manilatimes.net')) {
                                target.src = "https://www.pna.gov.ph/uploads/photos/2023/04/OCD-NDRRMC.jpg";
                              } else {
                                // Final fallback
                                target.src = "https://www.pagasa.dost.gov.ph/images/bulletin-images/satellite-images/himawari-visible.jpg";
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
      </Container>

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
        `}
      </style>
    </div>
  );
}