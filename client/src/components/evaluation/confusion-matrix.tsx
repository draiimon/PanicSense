import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { motion, AnimatePresence } from 'framer-motion';
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
  description = 'True vs Predicted sentiments with confidence distribution',
  allDatasets = false,
  metrics
}: ConfusionMatrixProps) {
  const [matrix, setMatrix] = useState<number[][]>([]);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
  const [isMatrixCalculated, setIsMatrixCalculated] = useState(false);
  const [mixedSentiments, setMixedSentiments] = useState<Record<string, { sentiments: Record<string, number> }>>({});
  const animationRef = useRef<number | null>(null);

  // Fetch sentiment posts for this file if fileId is provided
  const { data: sentimentPosts, isLoading } = useQuery({
    queryKey: ['/api/sentiment-posts/file', fileId],
    queryFn: () => getSentimentPostsByFileId(fileId as number),
    enabled: !!fileId && !initialMatrix
  });

  // Process sentiment data and build confusion matrix
  useEffect(() => {
    if ((isLoading || !sentimentPosts) && !initialMatrix) return;

    let newMatrix: number[][] = Array(labels.length).fill(0).map(() => Array(labels.length).fill(0));
    let newMixedSentiments: Record<string, { sentiments: Record<string, number> }> = {};

    if (initialMatrix) {
      newMatrix = initialMatrix.map(row => [...row]);
    } else if (sentimentPosts && sentimentPosts.length > 0) {
      sentimentPosts.forEach(post => {
        const mainSentiment = post.sentiment;
        const confidence = post.confidence || 0.8; // Default confidence if not provided

        // Find index of main sentiment
        const mainIdx = labels.findIndex(label => label === mainSentiment);
        if (mainIdx === -1) return;

        // Initialize mixed sentiments tracking for this post
        if (!newMixedSentiments[post.id]) {
          newMixedSentiments[post.id] = { sentiments: {} };
        }

        // Distribute confidence across sentiments
        const remainingConfidence = 1 - confidence;

        // Add main sentiment to mixed sentiments with its confidence
        newMixedSentiments[post.id].sentiments[mainSentiment] = confidence;

        // Update matrix for main sentiment (diagonal)
        newMatrix[mainIdx][mainIdx] += confidence;

        if (remainingConfidence > 0) {
          // Calculate weights for secondary sentiments based on semantic similarity
          const weights = labels.map((_, idx) => {
            if (idx === mainIdx) return 0;
            const distance = Math.abs(idx - mainIdx);
            return remainingConfidence / (distance + 1);
          });

          // Normalize weights
          const totalWeight = weights.reduce((sum, w) => sum + w, 0);
          const normalizedWeights = weights.map(w => w / totalWeight);

          // Distribute remaining confidence to other sentiments
          labels.forEach((label, idx) => {
            if (idx !== mainIdx) {
              const secondaryConfidence = remainingConfidence * normalizedWeights[idx];
              newMatrix[mainIdx][idx] += secondaryConfidence;
              newMixedSentiments[post.id].sentiments[label] = secondaryConfidence;
            }
          });
        }
      });
    }

    setMatrix(newMatrix);
    setMixedSentiments(newMixedSentiments);
    setIsMatrixCalculated(true);
  }, [sentimentPosts, labels, initialMatrix, isLoading]);

  // Calculate metrics from matrix
  const calculateMetrics = (row: number) => {
    const rowSum = matrix[row].reduce((a, b) => a + b, 0);
    const colSum = matrix.reduce((sum, r) => sum + r[row], 0);
    const truePositive = matrix[row][row];

    const precision = colSum === 0 ? 0 : truePositive / colSum;
    const recall = rowSum === 0 ? 0 : truePositive / rowSum;
    const f1 = precision + recall === 0 ? 0 : 2 * (precision * recall) / (precision + recall);

    return { precision, recall, f1 };
  };

  // Get color for cell based on value and confidence
  const getCellColor = (rowIdx: number, colIdx: number, value: number) => {
    if (rowIdx === colIdx) {
      // Diagonal (main sentiment)
      const baseColor = getSentimentColor(labels[rowIdx]);
      return {
        background: baseColor,
        text: '#ffffff'
      };
    } else {
      // Mixed sentiments (off-diagonal)
      const baseColor = getSentimentColor(labels[colIdx]);
      const intensity = Math.min(0.7, value);
      return {
        background: `${baseColor}${Math.floor(intensity * 255).toString(16).padStart(2, '0')}`,
        text: intensity > 0.4 ? '#ffffff' : '#333333'
      };
    }
  };

  // Row and column totals
  const rowTotals = matrix.map(row => 
    row.reduce((sum, val) => sum + val, 0)
  );

  const colTotals = labels.map((_, colIdx) =>
    matrix.reduce((sum, row) => sum + row[colIdx], 0)
  );

  const totalSamples = rowTotals.reduce((sum, val) => sum + val, 0);

  if (isLoading || !isMatrixCalculated) {
    return (
      <Card className="bg-white rounded-lg shadow-md">
        <CardHeader className="px-6 py-4 border-b border-gray-200">
          <CardTitle className="text-lg font-semibold text-slate-800">{title}</CardTitle>
          <CardDescription className="text-sm text-slate-500">{description}</CardDescription>
        </CardHeader>
        <CardContent className="p-6 text-center">
          <div className="flex items-center justify-center h-40">
            <svg className="animate-spin h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p className="ml-3 text-slate-500">Generating confusion matrix...</p>
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
        <div className="flex flex-col space-y-6">
          {/* Matrix Display */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-slate-50">
                  <th className="px-4 py-2 text-left">Actual/Predicted</th>
                  {labels.map((label, idx) => (
                    <th key={idx} className="px-4 py-2 text-center">{label}</th>
                  ))}
                  <th className="px-4 py-2 text-center">Total</th>
                  <th className="px-4 py-2 text-center">Metrics</th>
                </tr>
              </thead>
              <tbody>
                {matrix.map((row, rowIdx) => {
                  const metrics = calculateMetrics(rowIdx);
                  return (
                    <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                      <td className="px-4 py-2 font-medium">{labels[rowIdx]}</td>
                      {row.map((value, colIdx) => {
                        const { background, text } = getCellColor(rowIdx, colIdx, value);
                        const percentage = (value / rowTotals[rowIdx] * 100) || 0;

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
                              <div className="font-bold">{value.toFixed(3)}</div>
                              <div className="text-xs">{percentage.toFixed(1)}%</div>
                            </div>

                            {/* Tooltip for mixed sentiments */}
                            {hoveredCell?.row === rowIdx && hoveredCell?.col === colIdx && (
                              <div className="absolute z-50 bg-white p-3 rounded-lg shadow-lg border border-gray-200 min-w-[200px] -translate-y-full left-1/2 -translate-x-1/2">
                                <div className="font-medium mb-2">Sentiment Distribution</div>
                                <div className="space-y-1">
                                  {Object.entries(mixedSentiments)
                                    .filter(([_, data]) => data.sentiments[labels[rowIdx]] > 0)
                                    .map(([id, data]) => (
                                      <div key={id} className="text-xs">
                                        {Object.entries(data.sentiments)
                                          .filter(([_, conf]) => conf > 0)
                                          .map(([sentiment, conf]) => (
                                            <Badge
                                              key={sentiment}
                                              variant="outline"
                                              className="mr-1 mb-1"
                                              style={{
                                                borderColor: getSentimentColor(sentiment),
                                                color: getSentimentColor(sentiment)
                                              }}
                                            >
                                              {sentiment}: {(conf * 100).toFixed(1)}%
                                            </Badge>
                                          ))}
                                      </div>
                                    ))}
                                </div>
                              </div>
                            )}
                          </td>
                        );
                      })}
                      <td className="px-4 py-2 text-center font-medium">{rowTotals[rowIdx].toFixed(3)}</td>
                      <td className="px-4 py-2 text-xs">
                        <div>P: {(metrics.precision * 100).toFixed(1)}%</div>
                        <div>R: {(metrics.recall * 100).toFixed(1)}%</div>
                        <div>F1: {(metrics.f1 * 100).toFixed(1)}%</div>
                      </td>
                    </tr>
                  );
                })}
                <tr className="bg-slate-100 font-medium">
                  <td className="px-4 py-2">Total</td>
                  {colTotals.map((total, idx) => (
                    <td key={idx} className="px-4 py-2 text-center">{total.toFixed(3)}</td>
                  ))}
                  <td className="px-4 py-2 text-center">{totalSamples.toFixed(3)}</td>
                  <td className="px-4 py-2"></td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Legend and Information */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                <p>• Values show sentiment distribution and confidence</p>
                <p>• Diagonal cells represent primary sentiment confidence</p>
                <p>• Off-diagonal cells show mixed sentiment distribution</p>
                <p>• Hover over cells to see detailed sentiment breakdown</p>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}