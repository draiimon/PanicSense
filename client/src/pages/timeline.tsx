import { useState } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { SentimentTimeline } from "@/components/timeline/sentiment-timeline";
import { KeyEvents } from "@/components/timeline/key-events";
import { format, subDays } from "date-fns";

export default function Timeline() {
  const { disasterEvents, sentimentPosts } = useDisasterContext();

  // Process sentiment posts to create timeline data
  const processTimelineData = () => {
    let dates: string[] = [];
    
    // Extract actual dates from sentiment posts
    if (sentimentPosts.length > 0) {
      // Extract unique dates from actual data with full year information
      const uniqueDates = new Set<string>();
      const dateObjectMap = new Map<string, Date>();
      
      sentimentPosts.forEach(post => {
        const postDateObj = new Date(post.timestamp);
        const postDateFormatted = format(postDateObj, "yyyy-MM-dd");
        const displayDate = format(postDateObj, "MMM dd, yyyy");
        uniqueDates.add(displayDate);
        dateObjectMap.set(displayDate, postDateObj);
      });
      
      // Convert to array and sort chronologically by actual date timestamps
      dates = Array.from(uniqueDates).sort((a, b) => {
        const dateA = dateObjectMap.get(a) || new Date();
        const dateB = dateObjectMap.get(b) || new Date();
        return dateA.getTime() - dateB.getTime();
      });
    } else {
      // Fallback to using the last 7 days if no posts
      dates = Array.from({ length: 7 }, (_, i) => {
        const date = subDays(new Date(), i);
        return format(date, "MMM dd, yyyy");
      }).reverse();
    }

    // Initialize datasets for each sentiment
    const sentiments = ["Panic", "Fear/Anxiety", "Disbelief", "Resilience", "Neutral"];
    
    // Define proper type for sentiment counts
    const sentimentCounts: Record<string, Record<string, number>> = {};

    // Initialize counts for each sentiment on each date
    dates.forEach(date => {
      sentimentCounts[date] = {};
      sentiments.forEach(sentiment => {
        sentimentCounts[date][sentiment] = 0;
      });
    });

    // Count sentiments for each date
    sentimentPosts.forEach(post => {
      const postDate = format(new Date(post.timestamp), "MMM dd, yyyy");
      if (sentimentCounts[postDate] && sentimentCounts[postDate][post.sentiment] !== undefined) {
        sentimentCounts[postDate][post.sentiment]++;
      }
    });

    // Convert counts to percentages and create datasets
    const datasets = sentiments.map(sentiment => {
      const data = dates.map(date => {
        const total = Object.values(sentimentCounts[date]).reduce(
          (sum: number, count: number) => sum + count, 
          0
        );
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
        description={sentimentPosts.length > 0 ? 
          `${timelineData.labels.length} date${timelineData.labels.length !== 1 ? 's' : ''} from actual data` : 
          "Last 7 days"}
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