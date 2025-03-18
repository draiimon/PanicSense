import { useQuery } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useEffect, useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { Clock, AlertTriangle, Database, Timer, BarChart4 } from "lucide-react";

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
    
  // Get gradient colors based on percentage
  const getGradientColors = () => {
    if (percentUsed >= 100) return 'from-red-600 to-red-500';
    if (percentUsed >= 80) return 'from-amber-500 to-orange-500';
    if (percentUsed >= 60) return 'from-yellow-500 to-amber-500';
    return 'from-green-500 to-emerald-600';
  };
  
  return (
    <Card className="overflow-hidden shadow-lg border-none hover:shadow-xl transition-all duration-300 group">
      <div className={`h-full flex flex-col relative overflow-hidden rounded-xl`}>
        {/* Gradient background */}
        <div className={`absolute inset-0 bg-gradient-to-br from-green-600 to-teal-600 opacity-90`}></div>
        
        {/* Pattern overlay */}
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmZmZmYiIGZpbGwtb3BhY2l0eT0iMC4wNSI+PHBhdGggZD0iTTM2IDM0djZoNnYtNmgtNnptNiA2djZoNnYtNmgtNnptLTEyIDBoNnY2aC02di02em0xMiAwaDZ2NmgtNnYtNnoiLz48L2c+PC9nPjwvc3ZnPg==')] opacity-20"></div>
        
        {/* Content */}
        <CardContent className="p-6 relative z-10 h-full flex flex-col">
          {isLoading ? (
            <div className="flex flex-col space-y-4 animate-pulse">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <div className="h-3 w-20 bg-white/20 rounded"></div>
                  <div className="h-6 w-16 bg-white/20 rounded"></div>
                </div>
                <div className="h-12 w-12 rounded-full bg-white/20"></div>
              </div>
              <div className="h-3 w-32 bg-white/20 rounded mt-2"></div>
            </div>
          ) : (
            <>
              {/* Header with icon */}
              <div className="flex justify-between items-start mb-4">
                <div className={`w-12 h-12 rounded-full bg-white/20 flex items-center justify-center shadow-lg`}>
                  <Database className="h-6 w-6 text-white" />
                </div>
                <p className="text-sm font-medium uppercase tracking-wider text-white/80">Daily Usage</p>
              </div>
              
              {/* Main usage display */}
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-white/70">Processed</span>
                  <span className="text-xl font-bold text-white flex items-center gap-1">
                    {stats?.used}
                    <span className="text-sm text-white/60">/ {stats?.limit}</span>
                  </span>
                </div>
                
                {/* Custom progress bar */}
                <div className="h-2 rounded-full bg-black/10 backdrop-blur-sm overflow-hidden mb-2">
                  <div 
                    className={`h-full rounded-full bg-gradient-to-r ${getGradientColors()} transition-all duration-500`}
                    style={{ width: `${percentUsed}%` }}
                  ></div>
                </div>
                
                {/* Additional stats */}
                <div className="grid grid-cols-2 gap-2 mt-1">
                  <div className="rounded-lg bg-white/10 p-2 backdrop-blur-sm">
                    <div className="flex items-center gap-1.5 text-white/60 text-xs font-medium mb-1">
                      <Clock className="h-3 w-3" />
                      <span>Reset Timer</span>
                    </div>
                    <div className="text-sm font-medium text-white">
                      {resetTimeFormatted}
                    </div>
                  </div>
                  
                  <div className="rounded-lg bg-white/10 p-2 backdrop-blur-sm">
                    <div className="flex items-center gap-1.5 text-white/60 text-xs font-medium mb-1">
                      <BarChart4 className="h-3 w-3" />
                      <span>Remaining</span>
                    </div>
                    <div className="text-sm font-medium text-white">
                      {stats?.remaining} rows
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Warning message */}
              {percentUsed >= 90 && (
                <div className="mt-3 p-2 rounded-lg bg-black/10 backdrop-blur-sm border border-white/10">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-amber-300 flex-shrink-0" />
                    <p className="text-xs text-white">
                      {percentUsed >= 100 
                        ? "Daily limit reached! Processing will resume tomorrow." 
                        : "Approaching daily limit. Large files may be truncated."}
                    </p>
                  </div>
                </div>
              )}
              
              {/* Note about persistence */}
              <div className="mt-auto pt-3">
                <p className="text-[10px] text-white/60 italic">
                  * Counter persists even after data deletion
                </p>
              </div>
            </>
          )}
        </CardContent>
      </div>
    </Card>
  );
}