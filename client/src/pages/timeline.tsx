import { useState } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { SentimentTimeline } from "@/components/timeline/sentiment-timeline";
import { KeyEvents } from "@/components/timeline/key-events";
import { format, subDays } from "date-fns";

export default function Timeline() {
  const { disasterEvents, sentimentPosts } = useDisasterContext();

  // Process sentiment posts to create timeline data
  const processTimelineData = () => {
    // Get the last 7 days
    const dates = Array.from({ length: 7 }, (_, i) => {
      const date = subDays(new Date(), i);
      return format(date, "MMM dd");
    }).reverse();

    // Initialize datasets for each sentiment
    const sentiments = ["Panic", "Fear/Anxiety", "Disbelief", "Resilience", "Neutral"];
    const sentimentCounts = {};

    // Initialize counts for each sentiment on each date
    dates.forEach(date => {
      sentimentCounts[date] = {};
      sentiments.forEach(sentiment => {
        sentimentCounts[date][sentiment] = 0;
      });
    });

    // Count sentiments for each date
    sentimentPosts.forEach(post => {
      const postDate = format(new Date(post.timestamp), "MMM dd");
      if (sentimentCounts[postDate] && sentimentCounts[postDate][post.sentiment] !== undefined) {
        sentimentCounts[postDate][post.sentiment]++;
      }
    });

    // Convert counts to percentages and create datasets
    const datasets = sentiments.map(sentiment => {
      const data = dates.map(date => {
        const total = Object.values(sentimentCounts[date]).reduce((sum: number, count: number) => sum + count, 0);
        return total > 0 ? (sentimentCounts[date][sentiment] / total) * 100 : 0;
      });

      return {
        label: sentiment,
        data
      };
    });

    return {
      labels: dates,
      datasets
    };
  };

  const timelineData = processTimelineData();

  return (
    <div className="space-y-6">
      {/* Timeline Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Sentiment Timeline</h1>
        <p className="mt-1 text-sm text-slate-500">Tracking emotion changes over time</p>
      </div>

      {/* Timeline Chart */}
      <SentimentTimeline 
        data={timelineData}
        title="Sentiment Evolution"
        description="Last 7 days"
      />

      {/* Key Events */}
      <KeyEvents 
        events={disasterEvents.map(event => ({
          ...event,
          description: event.description || '' // Convert null to empty string
        }))}
        title="Key Events"
        description="Major shifts in sentiment patterns"
      />
    </div>
  );
}
import { useDisasterContext } from "@/context/disaster-context";
import { Card, CardContent } from "@/components/ui/card";

export default function Timeline() {
  const { sentimentPosts } = useDisasterContext();

  // Sort posts by date
  const sortedPosts = [...sentimentPosts].sort((a, b) => 
    new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Timeline</h1>
      
      <div className="space-y-4">
        {sortedPosts.map(post => (
          <Card key={post.id} className="relative">
            <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-500 rounded-l" />
            <CardContent className="p-4">
              <div className="flex justify-between mb-2">
                <span className="font-medium">{post.disasterType || 'Unknown Disaster'}</span>
                <span className="text-sm text-muted-foreground">
                  {new Date(post.timestamp).toLocaleString()}
                </span>
              </div>
              <p className="text-sm mb-2">{post.text}</p>
              <div className="flex gap-2">
                <span className={`text-xs px-2 py-1 rounded ${
                  post.sentiment === 'Panic' ? 'bg-red-100' :
                  post.sentiment === 'Fear/Anxiety' ? 'bg-orange-100' :
                  post.sentiment === 'Disbelief' ? 'bg-yellow-100' :
                  post.sentiment === 'Resilience' ? 'bg-green-100' :
                  'bg-gray-100'
                }`}>
                  {post.sentiment}
                </span>
                {post.location && (
                  <span className="text-xs px-2 py-1 bg-blue-100 rounded">
                    {post.location}
                  </span>
                )}
                <span className="text-xs px-2 py-1 bg-purple-100 rounded">
                  Confidence: {(post.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
