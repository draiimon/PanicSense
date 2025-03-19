import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { getSentimentPostsByFileId } from '@/lib/api';
import { getSentimentColor } from '@/lib/colors';
import { MetricsData } from './metrics-display';
import { Badge } from '@/components/ui/badge';

interface ConfusionMatrixProps {
  fileId?: number;
  confusionMatrix?: number[][];
  labels?: string[];
  title?: string;
  description?: string;
  allDatasets?: boolean;
  metrics?: MetricsData;
}

const defaultLabels = ['Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral'];

export function ConfusionMatrix({
  fileId,
  confusionMatrix: initialMatrix,
  labels = defaultLabels,
  title = 'Confusion Matrix',
  description = 'Real sentiment distribution based on data analysis',
  allDatasets = false,
  metrics
}: ConfusionMatrixProps) {
  const [matrix, setMatrix] = useState<number[][]>([]);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
  const [isMatrixCalculated, setIsMatrixCalculated] = useState(false);
  const [sentimentData, setSentimentData] = useState<{
    id: number;
    mainSentiment: string;
    confidence: number;
    mixedSentiments: Record<string, number>;
  }[]>([]);

  // Fetch sentiment posts if fileId is provided
  const { data: sentimentPosts, isLoading } = useQuery({
    queryKey: ['/api/sentiment-posts/file', fileId],
    queryFn: () => getSentimentPostsByFileId(fileId as number),
    enabled: !!fileId && !initialMatrix
  });

  // Process sentiment data and build confusion matrix
  useEffect(() => {
    if ((isLoading || !sentimentPosts) && !initialMatrix) return;

    // Initialize matrix with zeros
    let newMatrix: number[][] = Array(labels.length).fill(0).map(() => Array(labels.length).fill(0));
    let newSentimentData: typeof sentimentData = [];

    if (initialMatrix) {
      // Use provided matrix directly
      newMatrix = initialMatrix.map(row => [...row]);
    } else if (sentimentPosts && sentimentPosts.length > 0) {
      // Create counts for each sentiment
      const sentimentCounts: Record<string, number> = {};
      labels.forEach(label => {
        sentimentCounts[label] = 0;
      });

      // First pass: Count total occurrences of each sentiment
      sentimentPosts.forEach(post => {
        const mainSentiment = post.sentiment;
        if (mainSentiment in sentimentCounts) {
          sentimentCounts[mainSentiment]++;
        }
      });

      // Second pass: Build the matrix and track mixed sentiments
      sentimentPosts.forEach(post => {
        const mainSentiment = post.sentiment;
        const mainIdx = labels.findIndex(label => label === mainSentiment);
        if (mainIdx === -1) return;

        const confidence = post.confidence || 1;

        // For primary sentiment, always count as 1 (real count)
        newMatrix[mainIdx][mainIdx]++;

        // Track mixed sentiments for visualization
        const mixedSentiments: Record<string, number> = {
          [mainSentiment]: confidence
        };

        // If confidence < 1, distribute remaining for visualization
        if (confidence < 1) {
          const remainingConfidence = 1 - confidence;

          // Calculate distribution weights
          const weights = labels.map((_, idx) => {
            if (idx === mainIdx) return 0;
            const distance = Math.abs(idx - mainIdx);
            return remainingConfidence / (distance + 1);
          });

          // Normalize weights
          const totalWeight = weights.reduce((sum, w) => sum + w, 0);
          const normalizedWeights = weights.map(w => w / totalWeight);

          // Distribute remaining confidence for mixed sentiment display
          labels.forEach((label, idx) => {
            if (idx !== mainIdx) {
              const secondaryConfidence = remainingConfidence * normalizedWeights[idx];
              mixedSentiments[label] = secondaryConfidence;
            }
          });
        }

        // Store sentiment data for hover details
        newSentimentData.push({
          id: post.id,
          mainSentiment,
          confidence,
          mixedSentiments
        });
      });
    }

    setMatrix(newMatrix);
    setSentimentData(newSentimentData);
    setIsMatrixCalculated(true);
  }, [sentimentPosts, labels, initialMatrix, isLoading]);

  // Calculate cell color based on value
  const getCellColor = (rowIdx: number, colIdx: number, value: number) => {
    const baseColor = getSentimentColor(labels[colIdx]);
    if (rowIdx === colIdx) {
      return { background: baseColor, text: '#ffffff' };
    }
    const opacity = Math.min(0.7, value / Math.max(...matrix[rowIdx], 1));
    return {
      background: `${baseColor}${Math.floor(opacity * 255).toString(16).padStart(2, '0')}`,
      text: opacity > 0.4 ? '#ffffff' : '#333333'
    };
  };

  // Calculate totals
  const rowTotals = matrix.map(row => row.reduce((sum, val) => sum + val, 0));
  const colTotals = labels.map((_, colIdx) => matrix.reduce((sum, row) => sum + row[colIdx], 0));
  const totalSamples = rowTotals.reduce((sum, val) => sum + val, 0);

  if (isLoading || !isMatrixCalculated) {
    return (
      <Card className="bg-white rounded-lg shadow-md">
        <CardContent className="p-6 text-center">
          <div className="flex items-center justify-center h-40">
            <svg className="animate-spin h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p className="ml-3 text-slate-500">Calculating confusion matrix...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-white rounded-lg shadow-md">
      <CardHeader className="px-6 py-4 border-b border-gray-200">
        <CardTitle className="text-lg font-semibold text-slate-800">{title}</CardTitle>
        <CardDescription className="text-sm text-slate-500">{description}</CardDescription>
      </CardHeader>
      <CardContent className="p-6">
        {/* Matrix Display */}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-slate-50">
                <th className="px-4 py-2 text-left">True Sentiment</th>
                {labels.map((label, idx) => (
                  <th key={idx} className="px-4 py-2 text-center">{label}</th>
                ))}
                <th className="px-4 py-2 text-center">Total</th>
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, rowIdx) => (
                <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                  <td className="px-4 py-2 font-medium">{labels[rowIdx]}</td>
                  {row.map((value, colIdx) => {
                    const { background, text } = getCellColor(rowIdx, colIdx, value);
                    const percentage = rowTotals[rowIdx] ? (value / rowTotals[rowIdx] * 100) : 0;

                    return (
                      <td
                        key={colIdx}
                        className="px-4 py-2 relative"
                        onMouseEnter={() => setHoveredCell({ row: rowIdx, col: colIdx })}
                        onMouseLeave={() => setHoveredCell(null)}
                      >
                        <div
                          className="rounded p-2 text-center"
                          style={{ backgroundColor: background, color: text }}
                        >
                          <div className="font-bold">{value}</div>
                          <div className="text-xs">{percentage.toFixed(1)}%</div>
                        </div>

                        {/* Tooltip */}
                        {hoveredCell?.row === rowIdx && hoveredCell?.col === colIdx && (
                          <div className="absolute z-50 bg-white p-3 rounded-lg shadow-lg border border-gray-200 min-w-[250px] -translate-y-full left-1/2 -translate-x-1/2">
                            <div className="font-medium mb-2">Sentiment Distribution</div>
                            <div className="space-y-2">
                              {sentimentData
                                .filter(data => data.mainSentiment === labels[rowIdx])
                                .slice(0, 5)
                                .map((post, i) => (
                                  <div key={post.id} className="text-xs">
                                    <div className="font-medium text-slate-700 mb-1">Sample {i + 1}</div>
                                    <div className="flex flex-wrap gap-1">
                                      {Object.entries(post.mixedSentiments).map(([sentiment, conf]) => (
                                        <Badge
                                          key={sentiment}
                                          variant="outline"
                                          className="text-xs"
                                          style={{
                                            borderColor: getSentimentColor(sentiment),
                                            color: getSentimentColor(sentiment)
                                          }}
                                        >
                                          {sentiment}: {(conf * 100).toFixed(1)}%
                                        </Badge>
                                      ))}
                                    </div>
                                  </div>
                                ))}
                            </div>
                          </div>
                        )}
                      </td>
                    );
                  })}
                  <td className="px-4 py-2 text-center font-medium">{rowTotals[rowIdx]}</td>
                </tr>
              ))}
              {/* Totals row */}
              <tr className="bg-slate-100 font-medium">
                <td className="px-4 py-2">Total</td>
                {colTotals.map((total, idx) => (
                  <td key={idx} className="px-4 py-2 text-center">{total}</td>
                ))}
                <td className="px-4 py-2 text-center">{totalSamples}</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Legend and Information */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-50 p-4 rounded-lg">
            <h3 className="text-sm font-medium mb-2">Sentiment Legend</h3>
            <div className="flex flex-wrap gap-2">
              {labels.map((label) => (
                <Badge
                  key={label}
                  variant="outline"
                  className="flex items-center gap-1"
                  style={{
                    borderColor: getSentimentColor(label),
                    color: getSentimentColor(label)
                  }}
                >
                  {label}
                </Badge>
              ))}
            </div>
          </div>

          <div className="bg-slate-50 p-4 rounded-lg">
            <h3 className="text-sm font-medium mb-2">Matrix Information</h3>
            <div className="space-y-1 text-sm">
              <p>• Numbers show actual sentiment counts from data</p>
              <p>• Main diagonal shows primary sentiment counts</p>
              <p>• Hover over cells to see confidence distribution</p>
              <p>• Totals show overall sentiment distribution</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}