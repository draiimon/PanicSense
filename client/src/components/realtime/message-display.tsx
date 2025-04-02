import React from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertCircle } from "lucide-react";

interface MessageDisplayProps {
  text: string;
  sentiment: string;
  confidence: number;
  timestamp: Date | string;
  language: string;
  disasterType?: string;
  location?: string;
}

export function MessageDisplay({
  text,
  sentiment,
  confidence,
  timestamp,
  language,
  disasterType,
  location
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
    <Card className="bg-white/80 backdrop-blur-sm p-3 rounded-lg border border-slate-200/60 shadow-sm">
      <CardContent className="p-0 space-y-2">
        <div className="flex justify-between items-start">
          <p className="text-sm text-slate-900 whitespace-pre-wrap break-words">
            {text}
          </p>
          <div className="flex items-center gap-2">
            <Badge className={getSentimentBadgeClasses(sentiment)}>
              {sentiment}
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
      </CardContent>
    </Card>
  );
}