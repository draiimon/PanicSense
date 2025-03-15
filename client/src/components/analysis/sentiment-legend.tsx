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
  selectedRegion?: {
    name: string;
    sentiments: { name: string; percentage: number }[];
  } | null;
  showRegionSelection?: boolean;
  colorBy?: 'sentiment' | 'disasterType';
}

export function SentimentLegend({ 
  mostAffectedAreas = [],
  selectedRegion = null,
  showRegionSelection = true,
  colorBy = 'sentiment'
}: SentimentLegendProps) {
  // Sentiment colors and labels
  const sentiments = [
    { name: 'Panic', color: '#ef4444' },
    { name: 'Fear/Anxiety', color: '#f97316' },
    { name: 'Disbelief', color: '#8b5cf6' },
    { name: 'Resilience', color: '#10b981' },
    { name: 'Neutral', color: '#6b7280' }
  ];

  // Disaster type colors and labels
  const disasterTypes = [
    { name: 'Flood', color: '#3b82f6' },     // Blue
    { name: 'Typhoon', color: '#6b7280' },   // Gray
    { name: 'Fire', color: '#f97316' },      // Orange
    { name: 'Volcanic Eruption', color: '#ef4444' },   // Red
    { name: 'Earthquake', color: '#92400e' }, // Brown
    { name: 'Landslide', color: '#78350f' }  // Dark Brown
  ];

  return (
    <Card className="bg-white shadow-md border-none h-full">
      <CardHeader className="p-4 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <Map className="h-5 w-5 text-blue-600" />
          <div>
            <CardTitle className="text-lg font-medium text-slate-800">
              Geographic Analysis
            </CardTitle>
            <CardDescription className="text-sm text-slate-500">
              {colorBy === 'sentiment' ? 'Sentiment distribution' : 'Disaster impact'} by region
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-4 space-y-4">
        {/* Active Legend */}
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
          <div className="flex items-center gap-2 mb-3">
            {colorBy === 'sentiment' ? (
              <>
                <PieChart className="h-4 w-4 text-blue-600" />
                <h3 className="text-sm font-medium text-slate-700">Sentiment Indicators</h3>
              </>
            ) : (
              <>
                <AlertTriangle className="h-4 w-4 text-amber-500" />
                <h3 className="text-sm font-medium text-slate-700">Disaster Type Indicators</h3>
              </>
            )}
          </div>
          <div className="grid grid-cols-2 gap-2 ml-1">
            {(colorBy === 'sentiment' ? sentiments : disasterTypes).map((item) => (
              <div key={item.name} className="flex items-center space-x-2">
                <div 
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-sm text-slate-600">{item.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Selected Region Details */}
        {selectedRegion && (
          <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
            <div className="flex items-center gap-2 mb-3">
              <Globe className="h-4 w-4 text-green-600" />
              <h3 className="text-sm font-medium text-slate-700">{selectedRegion.name} Details</h3>
            </div>
            <div className="space-y-2">
              {selectedRegion.sentiments.map((sentiment) => (
                <div key={sentiment.name} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: getSentimentColor(sentiment.name) }}
                    />
                    <span className="text-sm text-slate-600">{sentiment.name}</span>
                  </div>
                  <span className="text-sm text-slate-600">{sentiment.percentage.toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Most Affected Areas */}
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              <h3 className="text-sm font-medium text-slate-700">Most Affected Areas</h3>
            </div>
            <Badge variant="outline" className="text-xs">Top {mostAffectedAreas.length}</Badge>
          </div>

          <div className="space-y-3">
            {mostAffectedAreas.length === 0 ? (
              <div className="flex items-center justify-center p-4 bg-white rounded-md border border-dashed border-slate-200">
                <p className="text-sm text-slate-500">No affected areas data available</p>
              </div>
            ) : (
              <div className="space-y-2">
                {mostAffectedAreas.map((area, index) => (
                  <div 
                    key={index} 
                    className="bg-white p-3 rounded-md border border-slate-200 hover:shadow-sm transition-shadow"
                  >
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-slate-800 flex items-center gap-1">
                        <Globe className="h-3.5 w-3.5 text-slate-500" />
                        {area.name}
                      </span>
                      <span className="text-xs text-slate-500">#{index + 1}</span>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      <Badge 
                        variant="outline"
                        className="text-xs flex items-center gap-1 border-2 font-medium"
                        style={{ 
                          borderColor: getSentimentColor(area.sentiment),
                          color: getSentimentColor(area.sentiment)
                        }}
                      >
                        <span 
                          className="w-2 h-2 rounded-full" 
                          style={{ backgroundColor: getSentimentColor(area.sentiment) }}
                        />
                        {area.sentiment}
                      </Badge>

                      {area.disasterType && (
                        <Badge
                          className="text-xs flex items-center gap-1 font-medium"
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
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}