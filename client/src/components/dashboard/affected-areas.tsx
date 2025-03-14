import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { getSentimentVariant } from '@/lib/colors';
import { useDisasterContext } from '@/context/disaster-context';
import { useMemo } from 'react';

interface AffectedArea {
  name: string;
  percentage: number;
  sentiment: string;
  count: number;
}

interface AffectedAreasProps {
  title?: string;
  description?: string;
}

export function AffectedAreas({ 
  title = 'Top Affected Areas',
  description = 'By sentiment intensity'
}: AffectedAreasProps) {
  const { sentimentPosts } = useDisasterContext();

  // Calculate affected areas from posts with explicit location data only
  const areas = useMemo(() => {
    const locationCounts = new Map<string, { total: number; sentiments: Map<string, number> }>();
    const totalPostsWithLocation = sentimentPosts.filter(post => post.location).length;

    if (totalPostsWithLocation === 0) return [];

    // Only process posts that have explicit location data
    sentimentPosts.forEach(post => {
      if (!post.location) return;

      const location = post.location;

      if (!locationCounts.has(location)) {
        locationCounts.set(location, {
          total: 0,
          sentiments: new Map()
        });
      }

      const locationData = locationCounts.get(location)!;
      locationData.total++;

      const currentSentimentCount = locationData.sentiments.get(post.sentiment) || 0;
      locationData.sentiments.set(post.sentiment, currentSentimentCount + 1);
    });

    // Convert to array and calculate percentages
    return Array.from(locationCounts.entries())
      .map(([location, data]) => {
        // Find dominant sentiment
        let maxCount = 0;
        let dominantSentiment = 'Neutral';

        data.sentiments.forEach((count, sentiment) => {
          if (count > maxCount) {
            maxCount = count;
            dominantSentiment = sentiment;
          }
        });

        return {
          name: location,
          count: data.total,
          percentage: (data.total / totalPostsWithLocation) * 100,
          sentiment: dominantSentiment
        };
      })
      .sort((a, b) => b.count - a.count)
      .slice(0, 5); // Get top 5 affected areas
  }, [sentimentPosts]);

  return (
    <Card className="bg-white rounded-lg shadow">
      <CardHeader className="p-5 border-b border-gray-200">
        <CardTitle className="text-lg font-medium text-slate-800">{title}</CardTitle>
        <CardDescription className="text-sm text-slate-500">{description}</CardDescription>
      </CardHeader>
      <CardContent className="p-5 space-y-4">
        {areas.length === 0 ? (
          <p className="text-center text-slate-500 py-4">No location data available</p>
        ) : (
          areas.map((area, index) => (
            <div key={index} className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-slate-700">{area.name}</span>
                <span className="text-sm text-slate-500">{Math.round(area.percentage)}%</span>
              </div>
              <Badge variant={getSentimentVariant(area.sentiment)}>
                {area.sentiment}
              </Badge>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}