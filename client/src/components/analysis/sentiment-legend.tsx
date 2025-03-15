import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { getSentimentColor, getDisasterTypeColor } from '@/lib/colors';

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
    <Card className="bg-white rounded-lg shadow">
      <CardHeader className="p-5 border-b border-gray-200">
        <CardTitle className="text-lg font-medium text-slate-800">
          Emotion Distribution
        </CardTitle>
        <CardDescription className="text-sm text-slate-500">
          By region and intensity
        </CardDescription>
      </CardHeader>
      <CardContent className="p-5 space-y-6">
        {/* Sentiment Legend */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-slate-700">Sentiment Colors</h3>
          {sentiments.map((sentiment) => (
            <div key={sentiment.name} className="flex items-center space-x-2">
              <div 
                className="w-4 h-4 rounded"
                style={{ backgroundColor: sentiment.color }}
              />
              <span className="text-sm text-slate-600">{sentiment.name}</span>
            </div>
          ))}
        </div>
        
        {/* Disaster Type Legend */}
        <div className="space-y-3 pt-4 border-t border-slate-200">
          <h3 className="text-sm font-medium text-slate-700">Disaster Type Colors</h3>
          {disasterTypes.map((disasterType) => (
            <div key={disasterType.name} className="flex items-center space-x-2">
              <div 
                className="w-4 h-4 rounded"
                style={{ backgroundColor: disasterType.color }}
              />
              <span className="text-sm text-slate-600">{disasterType.name}</span>
            </div>
          ))}
        </div>
        
        {/* Selected Region Info */}
        <div className="pt-4 border-t border-slate-200">
          <h3 className="text-sm font-medium text-slate-700 mb-3">
            Selected Region: <span className="font-bold">{selectedRegion?.name || 'None'}</span>
          </h3>
          <div className="space-y-2">
            {!selectedRegion ? (
              <p className="text-sm text-slate-500">Select a region on the map to view details</p>
            ) : (
              selectedRegion.sentiments.map((sentiment, index) => (
                <div key={index} className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">{sentiment.name}</span>
                  <span className="text-sm font-medium">{sentiment.percentage}%</span>
                </div>
              ))
            )}
          </div>
        </div>
        
        {/* Most Affected Areas */}
        <div className="pt-4 border-t border-slate-200">
          <h3 className="text-sm font-medium text-slate-700 mb-3">Most Affected Areas</h3>
          <div className="space-y-3">
            {mostAffectedAreas.length === 0 ? (
              <p className="text-sm text-slate-500">No data available</p>
            ) : (
              mostAffectedAreas.map((area, index) => (
                <div key={index} className="space-y-1">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-slate-700">{area.name}</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    <Badge 
                      variant={getSentimentVariant(area.sentiment) as any}
                    >
                      {area.sentiment}
                    </Badge>
                    
                    {area.disasterType && (
                      <Badge
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
              ))
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
