import { useState } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { DisasterComparison } from "@/components/comparison/disaster-comparison";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { SentimentChart } from "@/components/dashboard/sentiment-chart";

export default function Comparison() {
  const { sentimentPosts, disasterEvents } = useDisasterContext();

  // Simulate disaster type data
  // In a real app, this would be derived from sentimentPosts grouped by disasterType
  const disasterData = [
    {
      type: "Earthquake",
      sentiments: [
        { label: "Panic", percentage: 45 },
        { label: "Fear/Anxiety", percentage: 30 },
        { label: "Disbelief", percentage: 15 },
        { label: "Resilience", percentage: 5 },
        { label: "Neutral", percentage: 5 }
      ]
    },
    {
      type: "Typhoon",
      sentiments: [
        { label: "Panic", percentage: 20 },
        { label: "Fear/Anxiety", percentage: 35 },
        { label: "Disbelief", percentage: 15 },
        { label: "Resilience", percentage: 20 },
        { label: "Neutral", percentage: 10 }
      ]
    },
    {
      type: "Flood",
      sentiments: [
        { label: "Panic", percentage: 25 },
        { label: "Fear/Anxiety", percentage: 40 },
        { label: "Disbelief", percentage: 20 },
        { label: "Resilience", percentage: 10 },
        { label: "Neutral", percentage: 5 }
      ]
    },
    {
      type: "Volcanic Eruption",
      sentiments: [
        { label: "Panic", percentage: 50 },
        { label: "Fear/Anxiety", percentage: 25 },
        { label: "Disbelief", percentage: 10 },
        { label: "Resilience", percentage: 5 },
        { label: "Neutral", percentage: 10 }
      ]
    }
  ];

  // Comparison metrics for timeline
  const timeComparisonData = {
    labels: ["Initial Phase", "Peak Phase", "Recovery Phase"],
    values: [50, 75, 25],
    title: "Sentiment Intensity by Disaster Phase",
    description: "How emotions evolve throughout disaster lifecycle"
  };

  return (
    <div className="space-y-6">
      {/* Comparison Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Comparison</h1>
        <p className="mt-1 text-sm text-slate-500">Analyzing sentiment distribution across different disasters</p>
      </div>

      {/* Disaster Comparison Chart */}
      <DisasterComparison 
        disasters={disasterData}
        title="Disaster Type Comparison"
        description="Sentiment distribution across different disasters"
      />

      {/* Additional comparison insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <SentimentChart 
          data={timeComparisonData}
          type="bar"
        />

        <Card className="bg-white rounded-lg shadow">
          <CardHeader className="p-5 border-b border-gray-200">
            <CardTitle className="text-lg font-medium text-slate-800">Key Insights</CardTitle>
            <CardDescription className="text-sm text-slate-500">
              Important observations from cross-disaster analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="p-5">
            <ul className="space-y-4">
              <li className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-sm text-slate-700">
                  <span className="font-medium">Earthquakes</span> trigger the highest levels of panic initially, but sentiment shifts to resilience faster than other disasters.
                </p>
              </li>
              <li className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-sm text-slate-700">
                  <span className="font-medium">Typhoons</span> and <span className="font-medium">Floods</span> show similar sentiment patterns, with fear/anxiety being the predominant emotion.
                </p>
              </li>
              <li className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-sm text-slate-700">
                  <span className="font-medium">Volcanic eruptions</span> have the longest-lasting disbelief sentiment, likely due to their rarity and catastrophic nature.
                </p>
              </li>
              <li className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-sm text-slate-700">
                  <span className="font-medium">Resilience</span> emerges fastest in frequently occurring disasters, suggesting adaptation to common threats.
                </p>
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}