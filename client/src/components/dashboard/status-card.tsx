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
  BrainCircuit, 
  CheckCircle,
  ArrowUp,
  ArrowDown,
  Zap,
  Loader2
} from 'lucide-react';
import { getDisasterTypeColor } from '@/lib/colors';

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
    case 'brain': return BrainCircuit;
    case 'check-circle': return CheckCircle;
    case 'typhoon': case 'storm': return Wind;
    case 'flood': return Droplets;
    case 'fire': return Flame;
    case 'landslide': return Mountain;
    case 'earthquake': return AlertTriangle;
    case 'volcano': case 'eruption': return Flame;
    case 'tsunami': return Waves;
    default: return Cloud;
  }
};

export function StatusCard({ title, value, icon, trend, isLoading = false }: StatusCardProps) {
  // Determine icon based on title if not provided
  let IconComponent;
  let iconColor = '';
  
  if (icon) {
    IconComponent = getIconComponent(icon);
    
    // Set colors based on icon/title
    if (icon === 'alert-triangle') iconColor = '#f43f5e';
    else if (icon === 'bar-chart') iconColor = '#3b82f6';
    else if (icon === 'brain') iconColor = '#8b5cf6';
    else if (icon === 'check-circle') iconColor = '#10b981';
    else iconColor = getDisasterTypeColor(title);
  } else {
    IconComponent = getIconComponent(title.toLowerCase());
    iconColor = getDisasterTypeColor(title);
  }

  return (
    <Card className="bg-white shadow-lg border-none rounded-xl overflow-hidden hover:shadow-xl transition-shadow duration-300">
      <CardContent className="p-6">
        {isLoading ? (
          <div className="flex flex-col space-y-4 animate-pulse">
            <div className="flex items-center justify-between">
              <div className="space-y-2">
                <div className="h-3 w-20 bg-slate-200 rounded"></div>
                <div className="h-6 w-16 bg-slate-200 rounded"></div>
              </div>
              <div className="h-12 w-12 rounded-lg bg-slate-200"></div>
            </div>
            <div className="h-3 w-32 bg-slate-200 rounded mt-2"></div>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between">
              <div className="space-y-1.5">
                <p className="text-sm font-medium text-slate-500">{title}</p>
                <p className="text-3xl font-bold text-slate-800 tracking-tight">{value}</p>
              </div>
              <div 
                className="p-3 rounded-xl"
                style={{
                  background: `${iconColor}15`,
                }}
              >
                <IconComponent className="h-6 w-6" style={{ color: iconColor }} />
              </div>
            </div>

            {trend && (
              <div className="mt-4 flex items-center">
                <div className={`flex items-center gap-1 text-xs font-medium ${
                  trend.isUpward === null 
                    ? 'text-slate-500' 
                    : trend.isUpward 
                      ? 'text-emerald-500' 
                      : 'text-rose-500'
                }`}>
                  {trend.isUpward !== null && (
                    trend.isUpward ? (
                      <ArrowUp className="h-3.5 w-3.5" />
                    ) : (
                      <ArrowDown className="h-3.5 w-3.5" />
                    )
                  )}
                  <span>{trend.value}</span>
                </div>
                <span className="ml-2 text-xs text-slate-400">{trend.label}</span>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}