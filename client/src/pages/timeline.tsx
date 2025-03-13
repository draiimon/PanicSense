import { useState } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { SentimentTimeline } from "@/components/timeline/sentiment-timeline";
import { KeyEvents } from "@/components/timeline/key-events";

export default function Timeline() {
  const { disasterEvents, sentimentPosts } = useDisasterContext();

  // Mock timeline data for sentiment evolution
  // In a real app, this would be derived from sentimentPosts grouped by date
  const timelineData = {
    labels: ["May 10", "May 11", "May 12", "May 13", "May 14", "May 15", "May 16"],
    datasets: [
      {
        label: "Panic",
        data: [45, 28, 20, 15, 10, 8, 5]
      },
      {
        label: "Fear/Anxiety",
        data: [30, 42, 35, 30, 25, 20, 15]
      },
      {
        label: "Disbelief",
        data: [15, 20, 18, 15, 12, 10, 8]
      },
      {
        label: "Resilience",
        data: [5, 8, 15, 25, 35, 40, 45]
      },
      {
        label: "Neutral",
        data: [5, 2, 12, 15, 18, 22, 27]
      }
    ]
  };

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
        events={disasterEvents}
        title="Key Events"
        description="Major shifts in sentiment patterns"
      />
    </div>
  );
}
