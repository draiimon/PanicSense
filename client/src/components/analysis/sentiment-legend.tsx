import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { getSentimentColor, getDisasterTypeColor } from '@/lib/colors';
import { PieChart, Map, AlertTriangle, Globe } from 'lucide-react';

interface SentimentLegendProps {
  mostAffectedAreas?: {
    name: string;
    sentiment: string;
    disasterType?: string | null;
  }[];
  showRegionSelection?: boolean;
}

export function SentimentLegend({ 
  mostAffectedAreas = [],
  showRegionSelection = true 
}: SentimentLegendProps) {
  // Sentiment indicators
  const sentiments = [
    { name: 'Panic', color: '#ef4444' },
    { name: 'Fear/Anxiety', color: '#f97316' },
    { name: 'Disbelief', color: '#8b5cf6' },
    { name: 'Resilience', color: '#10b981' },
    { name: 'Neutral', color: '#6b7280' }
  ];

  // Disaster types
  const disasterTypes = [
    { name: 'Flood', color: '#3b82f6' },
    { name: 'Typhoon', color: '#6b7280' },
    { name: 'Fire', color: '#f97316' },
    { name: 'Volcano', color: '#ef4444' },
    { name: 'Earthquake', color: '#92400e' },
    { name: 'Landslide', color: '#78350f' }
  ];

  return (
    <Card className="bg-white shadow-md border-none h-full flex flex-col">
      <CardHeader className="p-4 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <Map className="h-5 w-5 text-blue-600" />
          <div>
            <CardTitle className="text-lg font-medium text-slate-800">
              Geographic Analysis
            </CardTitle>
            <CardDescription className="text-sm text-slate-500">
              Disaster impacts by region
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-4 space-y-4 flex-grow overflow-hidden">
        {/* Sentiment Legend */}
        <div className="bg-slate-50 p-3 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <PieChart className="h-4 w-4 text-blue-600" />
            <h3 className="text-sm font-medium text-slate-700">Sentiment Indicators</h3>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {sentiments.map((sentiment) => (
              <div key={sentiment.name} className="flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{ backgroundColor: sentiment.color }}
                />
                <span className="text-sm text-slate-600 truncate">{sentiment.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Disaster Types */}
        <div className="bg-slate-50 p-3 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="h-4 w-4 text-amber-500" />
            <h3 className="text-sm font-medium text-slate-700">Disaster Types</h3>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {disasterTypes.map((type) => (
              <div key={type.name} className="flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{ backgroundColor: type.color }}
                />
                <span className="text-sm text-slate-600 truncate">{type.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Affected Areas - With dedicated scrolling */}
        {mostAffectedAreas && mostAffectedAreas.length > 0 && (
          <div className="bg-slate-50 p-3 rounded-lg flex flex-col overflow-hidden">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Globe className="h-4 w-4 text-red-500" />
                <h3 className="text-sm font-medium text-slate-700">Most Affected Areas</h3>
              </div>
              <Badge variant="outline" className="text-xs">
                Top {mostAffectedAreas.length}
              </Badge>
            </div>
            <div className="space-y-2 overflow-y-auto max-h-[250px] pr-1 scrollbar-thin flex-grow">
              {mostAffectedAreas.map((area, index) => (
                <div 
                  key={index}
                  className="bg-white p-2 rounded-md border border-slate-200"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-slate-800 truncate max-w-[70%]">
                      {area.name}
                    </span>
                    <span className="text-xs text-slate-500">#{index + 1}</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    <Badge 
                      variant="outline"
                      className="text-xs"
                      style={{ 
                        borderColor: getSentimentColor(area.sentiment),
                        color: getSentimentColor(area.sentiment)
                      }}
                    >
                      {area.sentiment}
                    </Badge>
                    {area.disasterType && (
                      <Badge
                        className="text-xs"
                        style={{
                          backgroundColor: getDisasterTypeColor(area.disasterType),
                          color: 'white'
                        }}
                      >
                        {area.disasterType}
                      </Badge>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}