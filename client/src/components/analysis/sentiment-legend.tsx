import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { getSentimentColor, getDisasterTypeColor } from '@/lib/colors';
import { PieChart, Map, AlertTriangle, Globe } from 'lucide-react';

interface MostAffectedArea {
  name: string;
  sentiment: string;
  disasterType?: string | null;
  coordinates?: [number, number];
}

interface SentimentLegendProps {
  mostAffectedAreas?: MostAffectedArea[];
  showRegionSelection?: boolean;
  onSentimentClick?: (sentiment: string) => void;
}

export function SentimentLegend({ 
  mostAffectedAreas = [],
  showRegionSelection = true,
  onAreaClick,
  onSentimentClick
}: SentimentLegendProps & {
  onAreaClick?: (coordinates: [number, number]) => void;
}) {
  // Sentiment indicators
  const sentiments = [
    { name: 'Panic', color: '#ef4444' },
    { name: 'Fear/Anxiety', color: '#f97316' },
    { name: 'Disbelief', color: '#8b5cf6' },
    { name: 'Resilience', color: '#10b981' },
    { name: 'Neutral', color: '#6b7280' }
  ];

  // Disaster types - standardized to use "Volcanic Eruption"
  const disasterTypes = [
    { name: 'Flood', color: '#3b82f6' },
    { name: 'Typhoon', color: '#6b7280' },
    { name: 'Fire', color: '#f97316' },
    { name: 'Volcanic Eruption', color: '#ef4444' }, // Standardized name
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
              Legend
            </CardTitle>
            <CardDescription className="text-sm text-slate-500">
              Disaster impacts by region
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-4 space-y-4 flex-grow overflow-y-auto">
        {/* Sentiment Legend */}
        <div className="bg-slate-50 p-3 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <PieChart className="h-4 w-4 text-blue-600" />
            <h3 className="text-sm font-medium text-slate-700">Sentiment Indicators</h3>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {sentiments.map((sentiment) => (
              <div 
                key={sentiment.name} 
                className="flex items-center gap-2 cursor-pointer hover:bg-slate-100 p-1 rounded"
                onClick={() => onSentimentClick?.(sentiment.name)}
              >
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

        {/* Affected Areas section without separate scrolling */}
        {mostAffectedAreas && mostAffectedAreas.length > 0 && (
          <div className="bg-slate-50 p-3 rounded-lg max-h-[600px] overflow-y-auto">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Globe className="h-4 w-4 text-red-500" />
                <h3 className="text-sm font-medium text-slate-700">Most Affected Areas</h3>
              </div>
              <Badge variant="outline" className="text-xs">
                Top 10 Areas
              </Badge>
            </div>
            <div className="space-y-2">
              {mostAffectedAreas.map((area, index) => {
                // Standardize disaster type display
                const displayDisasterType = area.disasterType === 'Volcano' ? 'Volcanic Eruption' : area.disasterType;

                return (
                  <div 
                    key={index}
                    onClick={() => area.coordinates && onAreaClick?.(area.coordinates)}
                    className="bg-white p-2 rounded-md border border-slate-200 hover:bg-slate-50 transition-colors cursor-pointer active:bg-slate-100"
                    title="Click to zoom to location"
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
                      {displayDisasterType && (
                        <Badge
                          className="text-xs"
                          style={{
                            backgroundColor: getDisasterTypeColor(displayDisasterType),
                            color: 'white'
                          }}
                        >
                          {displayDisasterType}
                        </Badge>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}