import { ReactNode } from 'react';
import { Card, CardContent } from '@/components/ui/card';

interface StatusCardProps {
  title: string;
  value: string | number;
  icon: ReactNode;
  iconBgColor: string;
  change?: {
    value: string;
    positive: boolean;
  };
}

export function StatusCard({ title, value, icon, iconBgColor, change }: StatusCardProps) {
  return (
    <Card className="bg-white rounded-lg shadow">
      <CardContent className="p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-slate-400">{title}</p>
            <p className="text-2xl font-bold text-slate-800">{value}</p>
          </div>
          <div className={`${iconBgColor} p-3 rounded-full`}>
            {icon}
          </div>
        </div>
        
        {change && (
          <div className="mt-2">
            <span className={`text-xs font-medium ${change.positive ? 'text-green-500' : 'text-red-500'} flex items-center`}>
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                className="h-3 w-3 mr-1" 
                viewBox="0 0 20 20" 
                fill="currentColor"
              >
                {change.positive ? (
                  <path 
                    fillRule="evenodd" 
                    d="M12 7a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0V8.414l-4.293 4.293a1 1 0 01-1.414 0L8 10.414l-4.293 4.293a1 1 0 01-1.414-1.414l5-5a1 1 0 011.414 0L11 10.586 14.586 7H12z" 
                    clipRule="evenodd" 
                  />
                ) : (
                  <path 
                    fillRule="evenodd" 
                    d="M12 13a1 1 0 100 2h5a1 1 0 001-1v-5a1 1 0 10-2 0v2.586l-4.293-4.293a1 1 0 00-1.414 0L8 9.586l-4.293-4.293a1 1 0 00-1.414 1.414l5 5a1 1 0 001.414 0L11 9.414 14.586 13H12z" 
                    clipRule="evenodd" 
                  />
                )}
              </svg>
              {change.value}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
