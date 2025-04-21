import { useState } from "react";
import { Loader, ArrowUpRight, AlertTriangle, Zap, Clock } from "lucide-react";
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

        {/* Featured News Carousel */}
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
                      <CarouselItem key={item.id || index} className="md:basis-1/2 lg:basis-1/3">
                        <div className="p-1">
                          <Card className="overflow-hidden h-full flex flex-col border border-white/20 bg-white/10 backdrop-blur-md">
                            <CardHeader className="pb-2 bg-white/20">
                              <div className="flex justify-between items-start gap-2">
                                <Badge 
                                  className={`${getDisasterTypeColor(item.disasterType)} border border-white/20`}
                                >
                                  {formatDisasterType(item.disasterType)}
                                </Badge>
                                <Badge variant="outline" className="flex items-center gap-1 text-xs text-white border-white/20">
                                  <Clock className="h-3 w-3" />
                                  {formatDate(item.timestamp)}
                                </Badge>
                              </div>
                              <CardTitle className="text-base line-clamp-2 mt-2 text-white">
                                {item.title}
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="py-2 flex-grow">
                              <p className="text-sm text-white/80 line-clamp-3">
                                {item.content}
                              </p>
                            </CardContent>
                            <CardFooter className="pt-2 flex justify-between items-center border-t border-white/20">
                              <div className="text-xs flex items-center gap-1 text-white">
                                <span className="font-medium">
                                  {item.source}
                                </span>
                              </div>
                              <Button 
                                size="sm" 
                                variant="ghost" 
                                className="h-7 w-7 p-0 rounded-full text-white hover:bg-white/20" 
                                asChild
                              >
                                <a href={item.url} target="_blank" rel="noopener noreferrer">
                                  <ArrowUpRight className="h-4 w-4" />
                                </a>
                              </Button>
                            </CardFooter>
                          </Card>
                        </div>
                      </CarouselItem>
                    ))}
                  </CarouselContent>
                  <CarouselPrevious className="bg-white/30 hover:bg-white/50 border-none text-white" />
                  <CarouselNext className="bg-white/30 hover:bg-white/50 border-none text-white" />
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

        {/* News Grid */}
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
                      <Card className="h-full flex flex-col hover:shadow-md transition-shadow border-indigo-100">
                        <CardHeader className="pb-2 bg-gradient-to-r from-indigo-50 to-purple-50">
                          <div className="flex justify-between items-start gap-2">
                            <Badge 
                              variant="secondary"
                              className={getDisasterTypeColor(item.disasterType)}
                            >
                              {formatDisasterType(item.disasterType)}
                            </Badge>
                            <Badge variant="outline" className="flex items-center gap-1 text-xs">
                              <Clock className="h-3 w-3" />
                              {formatDate(item.timestamp)}
                            </Badge>
                          </div>
                          <CardTitle className="text-base mt-2">{item.title}</CardTitle>
                          <CardDescription className="mt-1">From {item.source}</CardDescription>
                        </CardHeader>
                        <CardContent className="py-3 flex-grow">
                          <p className="text-sm text-muted-foreground">{item.content}</p>
                        </CardContent>
                        <CardFooter className="pt-2 flex justify-between items-center bg-indigo-50/30">
                          <div className="text-xs font-medium text-indigo-800">{formatLocation(item.location)}</div>
                          <Button 
                            size="sm" 
                            variant="ghost" 
                            className="h-7 w-7 p-0 rounded-full text-indigo-600 hover:text-indigo-700 hover:bg-indigo-100" 
                            asChild
                          >
                            <a href={item.url} target="_blank" rel="noopener noreferrer">
                              <ArrowUpRight className="h-4 w-4" />
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
      <style jsx>{`
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
      `}</style>
    </div>
  );
}