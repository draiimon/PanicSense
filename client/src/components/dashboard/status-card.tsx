import { ReactNode } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Cloud, Droplets, Flame, Mountain, AlertTriangle, Wind, Waves } from 'lucide-react';
import { getDisasterTypeColor } from '@/lib/colors';

interface StatusCardProps {
  title: string;
  value: string | number;
  trend?: {
    value: string;
    isUpward: boolean | null;
    label: string;
  };
}

const getIconForDisaster = (title: string) => {
  const type = title.toLowerCase();
  if (type.includes('typhoon') || type.includes('storm')) return Wind;
  if (type.includes('flood')) return Droplets;
  if (type.includes('fire')) return Flame;
  if (type.includes('landslide')) return Mountain;
  if (type.includes('earthquake')) return AlertTriangle;
  if (type.includes('volcano') || type.includes('eruption')) return Flame;
  if (type.includes('tsunami')) return Waves;
  return Cloud;
};

export function StatusCard({ title, value, trend }: StatusCardProps) {
  const Icon = getIconForDisaster(title);
  const color = getDisasterTypeColor(title);

  const getIconBackground = () => {
    const baseColor = color.replace('#', '');
    // Create lighter version for background
    return {
      background: `#${baseColor}15`,
      color: color
    };
  };


  return (
    <Card className="bg-white/50 backdrop-blur-sm border-none shadow-sm hover:shadow-md transition-shadow duration-200">
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium text-slate-500">{title}</p>
            <p className="text-2xl font-bold text-slate-800">{value}</p>
          </div>
          <div 
            className="p-3 rounded-lg" 
            style={getIconBackground()}
          >
            <Icon className="h-6 w-6" />
          </div>
        </div>

        {trend && (
          <div className="mt-4 flex items-center">
            <span className={`text-xs font-medium flex items-center ${
              trend.isUpward === null 
                ? 'text-slate-500' 
                : trend.isUpward 
                  ? 'text-green-500' 
                  : 'text-red-500'
            }`}>
              {trend.isUpward !== null && (
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  className="h-3 w-3 mr-1" 
                  viewBox="0 0 20 20" 
                  fill="currentColor"
                >
                  {trend.isUpward ? (
                    <path 
                      fillRule="evenodd" 
                      d="M5.293 7.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 5.414V17a1 1 0 11-2 0V5.414L6.707 7.707a1 1 0 01-1.414 0z" 
                      clipRule="evenodd" 
                    />
                  ) : (
                    <path 
                      fillRule="evenodd" 
                      d="M14.707 12.293a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 14.586V3a1 1 0 012 0v11.586l2.293-2.293a1 1 0 011.414 0z" 
                      clipRule="evenodd" 
                    />
                  )}
                </svg>
              )}
              {trend.value}
            </span>
            <span className="ml-2 text-xs text-slate-400">{trend.label}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}