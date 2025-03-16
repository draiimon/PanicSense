import { ReactNode } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { AlertTriangle, BarChart2, Brain, CheckCircle, MessageSquare, MapPin, Activity } from 'lucide-react';

interface StatusCardProps {
  title: string;
  value: string | number;
  type: "disaster" | "posts" | "location" | "confidence" | "default"; // Changed to use 'type' instead of 'icon'
  trend?: {
    value: string;
    isUpward: boolean | null;
    label: string;
  };
}

export function StatusCard({ title, value, type = "default", trend }: StatusCardProps) {
  const getIconBackground = () => {
    switch (type) {
      case "disaster":
        return "bg-red-100 text-red-600";
      case "posts":
        return "bg-emerald-100 text-emerald-600";
      case "location":
        return "bg-purple-100 text-purple-600";
      case "confidence":
        return "bg-blue-100 text-blue-600";
      default:
        return "bg-slate-100 text-slate-600";
    }
  };

  const getIcon = () => {
    switch (type) {
      case "disaster":
        return <AlertTriangle className="h-6 w-6" />;
      case "posts":
        return <MessageSquare className="h-6 w-6" />;
      case "location":
        return <MapPin className="h-6 w-6" />;
      case "confidence":
        return <BarChart2 className="h-6 w-6" />;
      default:
        return <Activity className="h-6 w-6" />;
    }
  };

  return (
    <Card className="bg-white/50 backdrop-blur-sm border-none shadow-sm hover:shadow-md transition-shadow duration-200">
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium text-slate-500">{title}</p>
            <p className="text-2xl font-bold text-slate-800">{value}</p>
          </div>
          <div className={`${getIconBackground()} p-3 rounded-lg`}>
            {getIcon()}
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