import { ReactNode } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { 
  Cloud, 
  Droplets, 
  Flame, 
  Mountain, 
  AlertTriangle, 
  Wind, 
  Waves, 
  BarChart, 
  CheckCircle,
  ArrowUp,
  ArrowDown,
  Zap,
  Loader2,
  Activity,
  Heart,
  TrendingUp,
  Info
} from 'lucide-react';
import { getDisasterTypeColor, getSentimentColor } from '@/lib/colors';

export interface StatusCardProps {
  title: string;
  value: string | number;
  icon?: string;
  trend?: {
    value: string;
    isUpward: boolean | null;
    label: string;
  };
  isLoading?: boolean;
}

const getIconComponent = (iconName: string) => {
  switch (iconName) {
    case 'alert-triangle': return AlertTriangle;
    case 'bar-chart': return BarChart;
    case 'activity': return Activity;
    case 'check-circle': return CheckCircle;
    case 'heart': return Heart;
    case 'trending-up': return TrendingUp;
    case 'typhoon': case 'storm': return Wind;
    case 'flood': return Droplets;
    case 'fire': return Flame;
    case 'landslide': return Mountain;
    case 'earthquake': return AlertTriangle;
    case 'volcano': case 'eruption': return Flame;
    case 'tsunami': return Waves;
    default: return Info;
  }
};

// Card color schemes
const colorSchemes = {
  activeDisasters: {
    bg: 'from-red-500 to-orange-500',
    iconBg: 'bg-white/20',
    iconColor: 'text-white',
    textColor: 'text-white',
    trendColor: 'text-white/80'
  },
  analyzedPosts: {
    bg: 'from-blue-500 to-indigo-600',
    iconBg: 'bg-white/20',
    iconColor: 'text-white',
    textColor: 'text-white',
    trendColor: 'text-white/80'
  },
  dominantSentiment: {
    bg: 'from-purple-500 to-pink-500',
    iconBg: 'bg-white/20',
    iconColor: 'text-white',
    textColor: 'text-white',
    trendColor: 'text-white/80'
  },
  default: {
    bg: 'from-gray-700 to-gray-800',
    iconBg: 'bg-white/20',
    iconColor: 'text-white',
    textColor: 'text-white',
    trendColor: 'text-white/80'
  }
};

export function StatusCard({ title, value, icon, trend, isLoading = false }: StatusCardProps) {
  // Determine icon and color scheme
  let IconComponent;
  let scheme;
  let hasCustomIcon = false;
  
  // Set color scheme and icon based on card type
  switch (title) {
    case 'Active Disasters':
      IconComponent = getIconComponent('alert-triangle');
      scheme = colorSchemes.activeDisasters;
      break;
    case 'Analyzed Posts':
      IconComponent = getIconComponent('bar-chart');
      scheme = colorSchemes.analyzedPosts;
      break;
    case 'Dominant Sentiment':
      IconComponent = getIconComponent('heart');
      scheme = colorSchemes.dominantSentiment;
      break;
    default:
      IconComponent = icon ? getIconComponent(icon) : getIconComponent('info');
      scheme = colorSchemes.default;
      hasCustomIcon = !!icon;
  }

  // No custom sentiment styling for Dominant Sentiment card
  // We're making all text white as requested
  let sentimentColor = '';

  return (
    <Card className="overflow-hidden shadow-lg border-none hover:shadow-xl transition-all duration-300 group">
      <div className={`h-full flex flex-col relative overflow-hidden rounded-xl`}>
        {/* Gradient background */}
        <div className={`absolute inset-0 bg-gradient-to-br ${scheme.bg} opacity-90`}></div>
        
        {/* Animated pattern overlay */}
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
              {/* Icon and title */}
              <div className="flex justify-between items-start mb-4">
                <div className={`w-12 h-12 rounded-full ${scheme.iconBg} flex items-center justify-center shadow-lg`}>
                  <IconComponent className={`h-6 w-6 ${scheme.iconColor}`} />
                </div>
                <p className={`text-sm font-medium uppercase tracking-wider opacity-80 ${scheme.textColor}`}>{title}</p>
              </div>
              
              {/* Value */}
              <div className="mt-2">
                <h3 
                  className={`text-3xl font-bold tracking-tight ${scheme.textColor}`}
                  style={sentimentColor ? { color: sentimentColor } : {}}
                >
                  {value}
                </h3>
              </div>
              
              {/* Trend */}
              {trend && (
                <div className="mt-auto pt-4">
                  <div className="flex items-center space-x-1">
                    {trend.isUpward !== null && (
                      <span className={`flex items-center ${trend.isUpward ? 'text-green-300' : 'text-red-300'}`}>
                        {trend.isUpward ? (
                          <ArrowUp className="h-3 w-3 mr-0.5" />
                        ) : (
                          <ArrowDown className="h-3 w-3 mr-0.5" />
                        )}
                        {trend.value}
                      </span>
                    )}
                    {trend.isUpward === null && (
                      <span className={`text-sm ${scheme.trendColor}`}>{trend.value}</span>
                    )}
                    <span className={`text-xs ${scheme.trendColor}`}>{trend.label}</span>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </div>
    </Card>
  );
}