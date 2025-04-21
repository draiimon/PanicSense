import { useEffect, useState } from "react";
import { Loader, ArrowUpRight, AlertTriangle, Zap, Clock } from "lucide-react";
import { motion } from "framer-motion";
import { useToast } from "@/hooks/use-toast";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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

interface SocialMediaPost {
  id: string;
  content: string;
  date: string;
  source: string;
  user: {
    username: string;
    displayname: string;
    verified: boolean;
  };
  hashtags: string[];
  location?: string;
  url: string;
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
  const diffSec = Math.round(diffMs / 1000);
  const diffMin = Math.round(diffSec / 60);
  const diffHour = Math.round(diffMin / 60);
  const diffDay = Math.round(diffHour / 24);

  if (diffSec < 60) return `${diffSec} sec ago`;
  if (diffMin < 60) return `${diffMin} min ago`;
  if (diffHour < 24) return `${diffHour} hr ago`;
  if (diffDay < 30) return `${diffDay} days ago`;
  
  return date.toLocaleDateString();
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

export default function NewsMonitoringPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("all");
  
  // Fetch news data from API
  const { data: newsData, isLoading: newsLoading } = useQuery({
    queryKey: ['/api/real-news/posts'],
    refetchInterval: 60000, // Refetch every minute
  });
  
  // Fetch social media data from API
  const { data: socialData, isLoading: socialLoading } = useQuery({
    queryKey: ['/api/social-media/posts'],
    refetchInterval: 30000, // Refetch every 30 seconds
  });
  
  // Combined feed for "All" tab
  const { data: combinedData, isLoading: combinedLoading } = useQuery({
    queryKey: ['/api/combined-feed'],
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  // Manually refresh the feeds
  const handleRefresh = () => {
    toast({
      title: "Refreshing feeds",
      description: "Getting the latest updates...",
    });
    
    queryClient.invalidateQueries({ queryKey: ['/api/real-news/posts'] });
    queryClient.invalidateQueries({ queryKey: ['/api/social-media/posts'] });
    queryClient.invalidateQueries({ queryKey: ['/api/combined-feed'] });
  };

  return (
    <div className="min-h-screen">
      <Container>
        <PageHeader
          heading="Real-Time News Monitoring"
          subheading="Monitor disaster-related news and social media updates in real-time from official and community sources"
          className="mb-6"
        >
          <Button onClick={handleRefresh} className="gap-2">
            <Zap className="h-4 w-4" />
            Refresh Now
          </Button>
        </PageHeader>

        {/* Featured News Carousel */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2 text-yellow-500" />
            Latest Disaster Updates
          </h2>
          
          {combinedLoading ? (
            <div className="flex justify-center py-12">
              <Loader className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : combinedData && combinedData.length > 0 ? (
            <Carousel className="w-full">
              <CarouselContent>
                {combinedData.slice(0, 5).map((item: NewsItem | SocialMediaPost, index: number) => (
                  <CarouselItem key={item.id || index} className="md:basis-1/2 lg:basis-1/3">
                    <div className="p-1">
                      <Card className="overflow-hidden h-full flex flex-col">
                        <CardHeader className="pb-2">
                          <div className="flex justify-between items-start gap-2">
                            <Badge 
                              variant="secondary"
                              className={getDisasterTypeColor('disasterType' in item ? item.disasterType : undefined)}
                            >
                              {formatDisasterType('disasterType' in item ? item.disasterType : undefined)}
                            </Badge>
                            <Badge variant="outline" className="flex items-center gap-1 text-xs">
                              <Clock className="h-3 w-3" />
                              {formatDate('timestamp' in item ? item.timestamp : item.date)}
                            </Badge>
                          </div>
                          <CardTitle className="text-base line-clamp-2 mt-2">
                            {'title' in item ? item.title : item.content.split(' ').slice(0, 8).join(' ') + '...'}
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="py-2 flex-grow">
                          <p className="text-sm text-muted-foreground line-clamp-3">
                            {'content' in item ? item.content : item.content}
                          </p>
                        </CardContent>
                        <CardFooter className="pt-2 flex justify-between items-center">
                          <div className="text-xs flex items-center gap-1">
                            <span className="font-medium">
                              {'source' in item ? item.source : item.user.displayname}
                            </span>
                            {!('source' in item) && item.user.verified && (
                              <Badge variant="outline" className="px-1 py-0 h-4 text-[10px]">Official</Badge>
                            )}
                          </div>
                          <Button 
                            size="sm" 
                            variant="ghost" 
                            className="h-7 w-7 p-0 rounded-full" 
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
              <CarouselPrevious />
              <CarouselNext />
            </Carousel>
          ) : (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>No updates</AlertTitle>
              <AlertDescription>
                There are currently no disaster-related updates available.
              </AlertDescription>
            </Alert>
          )}
        </div>

        {/* Tabs for different feed types */}
        <Tabs 
          defaultValue="all" 
          value={activeTab}
          onValueChange={setActiveTab}
          className="w-full space-y-6"
        >
          <TabsList className="mb-4">
            <TabsTrigger value="all">All Updates</TabsTrigger>
            <TabsTrigger value="news">News Sources</TabsTrigger>
            <TabsTrigger value="social">Social Media</TabsTrigger>
          </TabsList>
          
          {/* All feeds combined */}
          <TabsContent value="all" className="space-y-4">
            {combinedLoading ? (
              <div className="flex justify-center py-12">
                <Loader className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : combinedData && combinedData.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {combinedData.map((item: NewsItem | SocialMediaPost, index: number) => (
                  <motion.div
                    key={item.id || index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                  >
                    <Card className="h-full flex flex-col hover:shadow-md transition-shadow">
                      <CardHeader className="pb-2">
                        <div className="flex justify-between items-start gap-2">
                          <Badge 
                            variant="secondary"
                            className={getDisasterTypeColor('disasterType' in item ? item.disasterType : undefined)}
                          >
                            {formatDisasterType('disasterType' in item ? item.disasterType : undefined)}
                          </Badge>
                          <Badge variant="outline" className="flex items-center gap-1 text-xs">
                            <Clock className="h-3 w-3" />
                            {formatDate('timestamp' in item ? item.timestamp : item.date)}
                          </Badge>
                        </div>
                        <CardTitle className="text-base mt-2">
                          {'title' in item ? item.title : `Update from ${item.user.displayname}`}
                        </CardTitle>
                        <CardDescription className="flex items-center mt-1">
                          {'source' in item ? (
                            <>From {item.source}</>
                          ) : (
                            <div className="flex items-center gap-1">
                              <span>@{item.user.username}</span>
                              {item.user.verified && (
                                <Badge variant="outline" className="px-1 py-0 h-4 text-[10px]">Official</Badge>
                              )}
                            </div>
                          )}
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="py-2 flex-grow">
                        <p className="text-sm text-muted-foreground">
                          {'content' in item ? item.content : item.content}
                        </p>
                        {!('title' in item) && item.hashtags && item.hashtags.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {item.hashtags.map((tag, idx) => (
                              <Badge key={idx} variant="secondary" className="text-xs px-1.5 py-0">
                                #{tag}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </CardContent>
                      <CardFooter className="pt-2 flex justify-between items-center">
                        <div className="text-xs">
                          {formatLocation('location' in item ? item.location : item.location)}
                        </div>
                        <Button 
                          size="sm" 
                          variant="ghost" 
                          className="h-7 w-7 p-0 rounded-full" 
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
              </div>
            ) : (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>No updates available</AlertTitle>
                <AlertDescription>
                  There are currently no disaster-related updates from any sources.
                </AlertDescription>
              </Alert>
            )}
          </TabsContent>
          
          {/* News sources tab */}
          <TabsContent value="news" className="space-y-4">
            {newsLoading ? (
              <div className="flex justify-center py-12">
                <Loader className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : newsData && newsData.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {newsData.map((item: NewsItem, index: number) => (
                  <motion.div
                    key={item.id || index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                  >
                    <Card className="h-full flex flex-col hover:shadow-md transition-shadow">
                      <CardHeader className="pb-2">
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
                      <CardContent className="py-2 flex-grow">
                        <p className="text-sm text-muted-foreground">{item.content}</p>
                      </CardContent>
                      <CardFooter className="pt-2 flex justify-between items-center">
                        <div className="text-xs">{formatLocation(item.location)}</div>
                        <Button 
                          size="sm" 
                          variant="ghost" 
                          className="h-7 w-7 p-0 rounded-full" 
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
              </div>
            ) : (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>No news available</AlertTitle>
                <AlertDescription>
                  There are currently no disaster-related news items available.
                </AlertDescription>
              </Alert>
            )}
          </TabsContent>
          
          {/* Social media tab */}
          <TabsContent value="social" className="space-y-4">
            {socialLoading ? (
              <div className="flex justify-center py-12">
                <Loader className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : socialData && socialData.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {socialData.map((post: SocialMediaPost, index: number) => (
                  <motion.div
                    key={post.id || index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                  >
                    <Card className="h-full flex flex-col hover:shadow-md transition-shadow">
                      <CardHeader className="pb-2">
                        <div className="flex justify-between items-start gap-2">
                          <div className="flex items-center gap-1">
                            <span className="font-medium">{post.user.displayname}</span>
                            {post.user.verified && (
                              <Badge variant="outline" className="px-1 py-0 h-4 text-[10px]">Official</Badge>
                            )}
                          </div>
                          <Badge variant="outline" className="flex items-center gap-1 text-xs">
                            <Clock className="h-3 w-3" />
                            {formatDate(post.date)}
                          </Badge>
                        </div>
                        <CardDescription className="mt-1">@{post.user.username}</CardDescription>
                      </CardHeader>
                      <CardContent className="py-2 flex-grow">
                        <p className="text-sm">{post.content}</p>
                        {post.hashtags && post.hashtags.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {post.hashtags.map((tag, idx) => (
                              <Badge key={idx} variant="secondary" className="text-xs px-1.5 py-0">
                                #{tag}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </CardContent>
                      <CardFooter className="pt-2 flex justify-between items-center">
                        <div className="text-xs">{formatLocation(post.location)}</div>
                        <Button 
                          size="sm" 
                          variant="ghost" 
                          className="h-7 w-7 p-0 rounded-full" 
                          asChild
                        >
                          <a href={post.url} target="_blank" rel="noopener noreferrer">
                            <ArrowUpRight className="h-4 w-4" />
                          </a>
                        </Button>
                      </CardFooter>
                    </Card>
                  </motion.div>
                ))}
              </div>
            ) : (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>No social media updates</AlertTitle>
                <AlertDescription>
                  There are currently no disaster-related social media posts available.
                </AlertDescription>
              </Alert>
            )}
          </TabsContent>
        </Tabs>
      </Container>
    </div>
  );
}