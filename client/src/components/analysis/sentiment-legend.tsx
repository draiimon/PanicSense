import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { getSentimentColor, getDisasterTypeColor } from '@/lib/colors';
import { PieChart, BarChart2, Map, AlertTriangle, Globe } from 'lucide-react';

interface SentimentLegendProps {
  selectedRegion?: {
    name: string;
    sentiments: {
      name: string;
      percentage: number;
    }[];
  };
  mostAffectedAreas?: {
    name: string;
    sentiment: string;
    disasterType?: string | null;
  }[];
}

export function SentimentLegend({ selectedRegion, mostAffectedAreas = [] }: SentimentLegendProps) {
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
    { name: 'Volcano', color: '#ef4444' },   // Red
    { name: 'Earthquake', color: '#92400e' },// Brown
    { name: 'Landslide', color: '#78350f' }  // Dark Brown
  ];

  // Get variant type for sentiment badge
  const getSentimentVariant = (sentiment: string) => {
    switch (sentiment) {
      case 'Panic': return 'panic';
      case 'Fear/Anxiety': return 'fear';
      case 'Disbelief': return 'disbelief';
      case 'Resilience': return 'resilience';
      case 'Neutral': 
      default: return 'neutral';
    }
  };

  return (
    <Card className="bg-white rounded-lg shadow-md border border-slate-200">
      <CardHeader className="p-5 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <BarChart2 className="h-5 w-5 text-blue-600" />
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
      <CardContent className="p-5 space-y-6">
        {/* Sentiment Legend */}
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
          <div className="flex items-center gap-2 mb-3">
            <PieChart className="h-4 w-4 text-blue-600" />
            <h3 className="text-sm font-medium text-slate-700">Sentiment Indicators</h3>
          </div>
          <div className="space-y-2 ml-1">
            {sentiments.map((sentiment) => (
              <div key={sentiment.name} className="flex items-center space-x-2">
                <div 
                  className="w-4 h-4 rounded-md shadow-sm"
                  style={{ backgroundColor: sentiment.color }}
                />
                <span className="text-sm text-slate-600">{sentiment.name}</span>
              </div>
            ))}
          </div>
        </div>
        
        {/* Disaster Type Legend */}
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="h-4 w-4 text-amber-500" />
            <h3 className="text-sm font-medium text-slate-700">Disaster Type Indicators</h3>
          </div>
          <div className="grid grid-cols-2 gap-2 ml-1">
            {disasterTypes.map((disasterType) => (
              <div key={disasterType.name} className="flex items-center space-x-2">
                <div 
                  className="w-4 h-4 rounded-md shadow-sm"
                  style={{ backgroundColor: disasterType.color }}
                />
                <span className="text-sm text-slate-600">{disasterType.name}</span>
              </div>
            ))}
          </div>
        </div>
        
        {/* Selected Region Info */}
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
          <div className="flex items-center gap-2 mb-3">
            <Map className="h-4 w-4 text-blue-600" />
            <h3 className="text-sm font-medium text-slate-700">
              Selected Region: <span className="font-semibold text-slate-900">{selectedRegion?.name || 'None'}</span>
            </h3>
          </div>
          
          <div className="space-y-2">
            {!selectedRegion ? (
              <div className="flex items-center justify-center p-4 bg-white rounded-md border border-dashed border-slate-200">
                <p className="text-sm text-slate-500 flex items-center gap-2">
                  <Globe className="h-4 w-4 text-slate-400" />
                  Select a region on the map to view details
                </p>
              </div>
            ) : (
              <div className="space-y-2 bg-white p-3 rounded-md border border-slate-200">
                {selectedRegion.sentiments.map((sentiment, index) => (
                  <div key={index} className="flex justify-between items-center p-1 hover:bg-slate-50 rounded">
                    <div className="flex items-center gap-2">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: getSentimentColor(sentiment.name) }}
                      />
                      <span className="text-sm text-slate-600">{sentiment.name}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="bg-slate-100 h-1.5 w-[50px] rounded-full overflow-hidden">
                        <div 
                          className="h-full rounded-full"
                          style={{ 
                            width: `${sentiment.percentage}%`,
                            backgroundColor: getSentimentColor(sentiment.name)
                          }}
                        ></div>
                      </div>
                      <span className="text-xs font-medium">{sentiment.percentage.toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
        
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
                        ></span>
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
