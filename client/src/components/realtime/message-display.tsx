import React from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, Shield, BrainCircuit } from "lucide-react";

interface MessageDisplayProps {
  text: string;
  sentiment: string;
  confidence: number;
  timestamp: Date | string;
  language: string;
  explanation?: string;
  aiTrustMessage?: string;
  disasterType?: string;
  location?: string;
  corrected?: boolean;
}

export function MessageDisplay({
  text,
  sentiment,
  confidence,
  timestamp,
  language,
  explanation,
  aiTrustMessage,
  disasterType,
  location,
  corrected = false
}: MessageDisplayProps) {
  // Helper function for sentiment badge styles
  const getSentimentBadgeClasses = (sentiment: string) => {
    switch (sentiment) {
      case 'Panic':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'Fear/Anxiety':
        return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'Disbelief':
        return 'bg-purple-100 text-purple-800 border-purple-200';
      case 'Resilience':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'Neutral':
        return 'bg-slate-100 text-slate-800 border-slate-200';
      default:
        return 'bg-slate-100 text-slate-800 border-slate-200';
    }
  };

  // Format timestamp
  const formatTime = (timestamp: Date | string) => {
    if (typeof timestamp === 'string') {
      return new Date(timestamp).toLocaleTimeString();
    }
    return timestamp.toLocaleTimeString();
  };

  return (
    <Card className={`bg-white/80 backdrop-blur-sm p-3 rounded-lg border ${corrected ? 'border-blue-300 shadow-blue-100/50' : 'border-slate-200/60'} shadow-sm`}>
      <CardContent className="p-0 space-y-2">
        <div className="flex justify-between items-start">
          <p className="text-sm text-slate-900 whitespace-pre-wrap break-words">
            {text}
          </p>
          <div className="flex items-center gap-2">
            <Badge className={getSentimentBadgeClasses(sentiment)}>
              {sentiment}
              {corrected && (
                <span className="ml-1 text-xs opacity-70">(✓)</span>
              )}
            </Badge>
            <Badge variant="outline" className="bg-slate-100">
              {language === "tl" ? "Filipino" : "English"}
            </Badge>
          </div>
        </div>

        <div className="flex flex-wrap justify-between items-center text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <span>Confidence: {(confidence * 100).toFixed(2)}%</span>
          </div>
          <span>{formatTime(timestamp)}</span>
        </div>

        {disasterType && disasterType !== "Not Specified" && disasterType !== "UNKNOWN" && (
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
              {disasterType}
            </Badge>
            {location && location !== "UNKNOWN" && (
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                {location}
              </Badge>
            )}
          </div>
        )}

        {/* AI Explanation Section */}
        {explanation && !explanation.includes("Fallback") && (
          <div className="bg-gradient-to-r from-blue-50/90 to-indigo-50/90 backdrop-blur-sm p-3 rounded-md border border-blue-200/50 shadow-sm">
            <div className="flex items-start gap-2">
              <BrainCircuit className="h-5 w-5 text-blue-600 mt-0.5" />
              <div className="w-full">
                <h4 className="text-sm font-medium mb-1 text-blue-800">Sentiment Analysis</h4>
                
                <div className="text-sm text-slate-700 p-2 bg-white/80 rounded border border-blue-100">
                  <span className="text-slate-700">{explanation}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* AI Trust Message / Validation Message */}
        {aiTrustMessage && (
          <div className="bg-gradient-to-r from-amber-50/90 to-orange-50/90 backdrop-blur-sm p-3 rounded-md border border-amber-200/50 shadow-sm">
            <div className="flex items-start gap-2">
              <Shield className="h-5 w-5 text-amber-600 mt-0.5" />
              <div className="w-full">
                <h4 className="text-sm font-medium mb-1 text-amber-800">Feedback Validation</h4>
                
                <div className="text-sm text-slate-700 p-2 bg-white/80 rounded border border-amber-100">
                  <span className="text-amber-800">{aiTrustMessage}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}