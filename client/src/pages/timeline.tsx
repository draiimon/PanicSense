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