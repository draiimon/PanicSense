import { useMemo } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { SentimentTimeline } from "@/components/timeline/sentiment-timeline";
import { KeyEvents } from "@/components/timeline/key-events";
import { format, isSameDay, parseISO } from "date-fns";

export default function Timeline() {
  const { disasterEvents, sentimentPosts } = useDisasterContext();

  // Process sentiment posts to create timeline data
  const processTimelineData = () => {
    // Skip processing if no posts
    if (!sentimentPosts || sentimentPosts.length === 0) {
      return {
        labels: [],
        datasets: [],
        rawDates: []
      };
    }
    
    // First, sort posts chronologically
    const sortedPosts = [...sentimentPosts].sort((a, b) => {
      return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
    });
    
    // Extract all raw dates
    const rawDates = sortedPosts.map(post => post.timestamp);
    
    // Track unique dates to display as labels
    const uniqueDates = new Map<string, Date>();
    
    // Group sentiment posts by date
    sortedPosts.forEach(post => {
      const postDate = parseISO(post.timestamp);
      const displayDate = format(postDate, "MMM dd, yyyy");
      uniqueDates.set(displayDate, postDate);
    });
    
    // Convert to sorted array of labels
    const dates = Array.from(uniqueDates.entries())
      .sort(([_, dateA], [__, dateB]) => dateA.getTime() - dateB.getTime())
      .map(([label]) => label);

    // Initialize datasets for each sentiment
    const sentiments = ["Panic", "Fear/Anxiety", "Disbelief", "Resilience", "Neutral"];
    
    // Track sentiment counts per date for percentage calculation
    const sentimentCounts: Record<string, Record<string, number>> = {};

    // Initialize counts
    dates.forEach(date => {
      sentimentCounts[date] = {};
      sentiments.forEach(sentiment => {
        sentimentCounts[date][sentiment] = 0;
      });
    });

    // Count sentiments for each date
    sortedPosts.forEach(post => {
      const postDate = format(parseISO(post.timestamp), "MMM dd, yyyy");
      if (sentimentCounts[postDate] && post.sentiment) {
        sentimentCounts[postDate][post.sentiment] = 
          (sentimentCounts[postDate][post.sentiment] || 0) + 1;
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
      datasets,
      rawDates
    };
  };

  // Calculate timeline data with memoization to avoid recalculation
  const timelineData = useMemo(() => processTimelineData(), [sentimentPosts]);
  
  // No auto-generation of events
  const shouldGenerateEvents = false;

  return (
    <div className="space-y-6">
      {/* Timeline Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Sentiment Timeline</h1>
        <p className="mt-1 text-sm text-slate-500">
          {sentimentPosts.length > 0 
            ? `Tracking ${sentimentPosts.length} data points across ${timelineData.labels.length} unique dates` 
            : "No data available yet. Upload a CSV file to begin analysis."}
        </p>
      </div>

      {/* Timeline Chart */}
      <SentimentTimeline 
        data={timelineData}
        title="Sentiment Evolution"
        rawDates={timelineData.rawDates}
      />

      {/* Key Events */}
      <KeyEvents 
        events={disasterEvents.map(event => ({
          ...event,
          description: event.description || '' // Convert null to empty string
        }))}
        title="Key Events"
        description="Major shifts in sentiment patterns"
        sentimentPosts={shouldGenerateEvents ? sentimentPosts : []}
      />
    </div>
  );
}