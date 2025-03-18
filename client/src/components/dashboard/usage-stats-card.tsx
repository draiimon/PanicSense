import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useEffect, useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { Clock, AlertTriangle, Database } from "lucide-react";

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
  
  // Determine color based on percentage
  const getProgressColor = () => {
    if (percentUsed >= 100) return "bg-red-500";
    if (percentUsed >= 80) return "bg-amber-500";
    if (percentUsed >= 60) return "bg-yellow-400";
    return "bg-blue-500";
  };
  
  // Keep count even after deletion
  const permanentCountMessage = stats?.used > 0 
    ? "Counter persists even if data is deleted"
    : "";
  
  return (
    <Card className="relative overflow-hidden">
      <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0 bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-blue-100/50">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-blue-500/10">
            <Database className="text-blue-600 h-5 w-5" />
          </div>
          <CardTitle className="text-lg font-semibold text-slate-800">Daily Usage</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="pt-6">
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
              <span className="text-gray-600 text-sm">Rows processed today</span>
              <span className={`font-semibold text-lg ${percentUsed >= 100 ? 'text-red-600' : 'text-gray-800'}`}>
                {stats?.used} <span className="text-gray-400 text-sm">/ {stats?.limit}</span>
              </span>
            </div>
            
            <div className="relative pt-1">
              <div className="overflow-hidden h-2 text-xs flex rounded bg-gray-200">
                <div 
                  className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center transition-all duration-500 ${getProgressColor()}`} 
                  style={{ width: `${percentUsed}%` }}
                ></div>
              </div>
            </div>
            
            <div className="flex items-center justify-between text-xs pt-1">
              <span className="text-gray-500 flex items-center">
                <Clock className="h-3 w-3 mr-1" />
                Resets {resetTimeFormatted}
              </span>
              <span className="text-gray-500">
                {stats?.remaining} rows remaining
              </span>
            </div>
            
            {stats?.used > 0 && (
              <div className="text-xs text-gray-500 italic mt-1">
                {permanentCountMessage}
              </div>
            )}
            
            {percentUsed >= 90 && (
              <div className="text-xs flex items-center gap-1 font-medium mt-2 bg-red-50 text-red-600 p-2 rounded-md border border-red-100">
                <AlertTriangle className="h-3 w-3" />
                {percentUsed >= 100 
                  ? "Daily limit reached! Processing will resume tomorrow." 
                  : "Approaching daily limit. Large files may be truncated."}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}