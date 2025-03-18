import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useEffect, useState } from "react";
import { formatDistanceToNow } from "date-fns";

interface UsageStats {
  used: number;
  limit: number;
  remaining: number;
  resetAt: string;
}

export function UsageStatsCard() {
  const [refreshInterval, setRefreshInterval] = useState<number | null>(null);
  
  // Get the usage stats
  const { data, isLoading, error } = useQuery({
    queryKey: ["/api/usage-stats"],
    refetchInterval: refreshInterval || false,
  });
  
  const stats: UsageStats = data as UsageStats;
  
  // Auto-refresh more frequently when a file is being processed
  useEffect(() => {
    // Set a default 1-minute refresh interval
    setRefreshInterval(60000);
    
    return () => {
      setRefreshInterval(null);
    };
  }, []);
  
  // Calculate percentage of limit used
  const percentUsed = stats ? Math.min(100, Math.round((stats.used / stats.limit) * 100)) : 0;
  
  // Format the reset time
  const resetTimeFormatted = stats?.resetAt 
    ? formatDistanceToNow(new Date(stats.resetAt), { addSuffix: true })
    : 'Unknown';
  
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">Daily API Usage</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="h-16 flex items-center justify-center">
            <p className="text-muted-foreground text-sm">Loading usage data...</p>
          </div>
        ) : error ? (
          <div className="h-16 flex items-center justify-center">
            <p className="text-red-500 text-sm">Failed to load usage data</p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground text-sm">Rows processed today</span>
              <span className="font-semibold">
                {stats?.used} / {stats?.limit}
              </span>
            </div>
            
            <Progress value={percentUsed} className="h-2" />
            
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">
                {stats?.remaining} rows remaining
              </span>
              <span className="text-muted-foreground">
                Resets {resetTimeFormatted}
              </span>
            </div>
            
            {percentUsed >= 90 && (
              <div className="text-xs text-amber-500 font-medium mt-2">
                {percentUsed >= 100 
                  ? "Daily limit reached. Processing will resume tomorrow." 
                  : "Approaching daily limit. Large files may be truncated."}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}