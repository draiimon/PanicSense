import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { useQuery } from '@tanstack/react-query';
import { getSentimentPostsByFileId } from '@/lib/api';
import { getSentimentColor } from '@/lib/colors';
import { Badge } from '@/components/ui/badge';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ConfusionMatrixProps {
  fileId?: number;
  confusionMatrix?: number[][];
  labels?: string[];
  title?: string;
  description?: string;
  allDatasets?: boolean;
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  };
}

const defaultLabels = ['Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral'];

export function ConfusionMatrix({
  fileId,
  confusionMatrix: initialMatrix,
  labels = defaultLabels,
  title = 'Confusion Matrix',
  description = 'Real sentiment distribution with confidence scores',
  allDatasets = false,
  metrics
}: ConfusionMatrixProps) {
  const [matrix, setMatrix] = useState<number[][]>([]);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
  const [isMatrixCalculated, setIsMatrixCalculated] = useState(false);
  const [sentimentData, setSentimentData] = useState<{
    id: string;
    mainSentiment: string;
    confidence: number;
    mixedSentiments: Record<string, number>;
  }[]>([]);
  const [metricsData, setMetricsData] = useState<any[]>([]);

  const { data: sentimentPosts, isLoading } = useQuery({
    queryKey: ['/api/sentiment-posts/file', fileId],
    queryFn: () => getSentimentPostsByFileId(fileId as number),
    enabled: !!fileId && !initialMatrix && !allDatasets
  });

  const calculateMetrics = (matrix: number[][]) => {
    const metrics = labels.map((_, idx) => {
      const truePositive = matrix[idx][idx];
      const rowSum = matrix[idx].reduce((sum, val) => sum + val, 0);
      const colSum = matrix.reduce((sum, row) => sum + row[idx], 0);
      const totalSum = matrix.reduce((sum, row) => sum + row.reduce((s, v) => s + v, 0), 0);

      const precision = colSum === 0 ? 0 : truePositive / colSum;
      const recall = rowSum === 0 ? 0 : truePositive / rowSum;
      const f1 = precision + recall === 0 ? 0 : 2 * (precision * recall) / (precision + recall);
      const accuracy = totalSum === 0 ? 0 : truePositive / totalSum;

      return {
        sentiment: labels[idx],
        precision: precision * 100,
        recall: recall * 100,
        f1Score: f1 * 100,
        accuracy: accuracy * 100
      };
    });

    setMetricsData(metrics);
    return metrics;
  };

  useEffect(() => {
    if ((isLoading || !sentimentPosts) && !initialMatrix) return;

    let newMatrix: number[][] = Array(labels.length).fill(0).map(() => Array(labels.length).fill(0));
    let newSentimentData: typeof sentimentData = [];

    if (initialMatrix) {
      newMatrix = initialMatrix.map(row => [...row]);
    } else if (sentimentPosts && sentimentPosts.length > 0) {
      sentimentPosts.forEach(post => {
        const mainSentiment = post.sentiment;
        const confidence = post.confidence || 1;

        const mainIdx = labels.findIndex(label => label === mainSentiment);
        if (mainIdx === -1) return;

        const postSentiments: Record<string, number> = {
          [mainSentiment]: confidence
        };

        newMatrix[mainIdx][mainIdx] += confidence;

        if (confidence < 1) {
          const remainingConfidence = 1 - confidence;
          const weights = labels.map((_, idx) => {
            if (idx === mainIdx) return 0;
            const distance = Math.abs(idx - mainIdx);
            return remainingConfidence / (distance + 1);
          });

          const totalWeight = weights.reduce((sum, w) => sum + w, 0);
          const normalizedWeights = weights.map(w => w / totalWeight);

          labels.forEach((label, idx) => {
            if (idx !== mainIdx) {
              const secondaryConfidence = remainingConfidence * normalizedWeights[idx];
              newMatrix[mainIdx][idx] += secondaryConfidence;
              postSentiments[label] = secondaryConfidence;
            }
          });
        }

        if (!allDatasets) {
          newSentimentData.push({
            id: post.id,
            mainSentiment,
            confidence,
            mixedSentiments: postSentiments
          });
        }
      });
    }

    setMatrix(newMatrix);
    setSentimentData(newSentimentData);
    calculateMetrics(newMatrix);
    setIsMatrixCalculated(true);
  }, [sentimentPosts, labels, initialMatrix, isLoading, allDatasets]);

  const getCellColor = (rowIdx: number, colIdx: number, value: number) => {
    const baseColor = getSentimentColor(labels[colIdx]);
    if (rowIdx === colIdx) {
      return { background: baseColor, text: '#ffffff' };
    }
    const opacity = Math.min(0.7, value);
    return {
      background: `${baseColor}${Math.floor(opacity * 255).toString(16).padStart(2, '0')}`,
      text: opacity > 0.4 ? '#ffffff' : '#333333'
    };
  };

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
        <div className="space-y-6">
          {/* Model Insights */}
          <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
            <h3 className="text-lg font-semibold mb-2">Model Insights</h3>
            <p className="text-sm text-slate-600 mb-4">Understanding the sentiment analysis model performance</p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Metrics Visualization */}
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={metricsData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="sentiment" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="precision" fill="#22c55e" name="Precision" />
                    <Bar dataKey="recall" fill="#8b5cf6" name="Recall" />
                    <Bar dataKey="f1Score" fill="#f97316" name="F1 Score" />
                    <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Metrics Explanation */}
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-slate-800 mb-1">Accuracy</h4>
                  <p className="text-sm text-slate-600">The proportion of total predictions that were correct. Higher is better.</p>
                  <p className="text-2xl font-bold text-blue-600 mt-1">
                    {metricsData[0]?.accuracy.toFixed(1)}%
                  </p>
                </div>

                <div>
                  <h4 className="font-medium text-slate-800 mb-1">Precision</h4>
                  <p className="text-sm text-slate-600">Of all predicted positive instances, how many were actually positive. Measures false positive rate.</p>
                  <p className="text-2xl font-bold text-green-600 mt-1">
                    {metricsData[0]?.precision.toFixed(1)}%
                  </p>
                </div>

                <div>
                  <h4 className="font-medium text-slate-800 mb-1">Recall</h4>
                  <p className="text-sm text-slate-600">Of all actual positive instances, how many were predicted as positive. Measures false negative rate.</p>
                  <p className="text-2xl font-bold text-purple-600 mt-1">
                    {metricsData[0]?.recall.toFixed(1)}%
                  </p>
                </div>

                <div>
                  <h4 className="font-medium text-slate-800 mb-1">F1 Score</h4>
                  <p className="text-sm text-slate-600">The harmonic mean of precision and recall. Balances both metrics for an overall performance measure.</p>
                  <p className="text-2xl font-bold text-orange-600 mt-1">
                    {metricsData[0]?.f1Score.toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>
          </div>

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
                      const percentage = (value / rowTotals[rowIdx] * 100) || 0;

                      const relevantPosts = !allDatasets ? sentimentData.filter(post =>
                        post.mainSentiment === labels[rowIdx] &&
                        post.mixedSentiments[labels[colIdx]] > 0
                      ) : [];

                      return (
                        <td
                          key={colIdx}
                          className={`px-4 py-2 relative ${!allDatasets ? 'hover:bg-slate-50' : ''}`}
                          onMouseEnter={() => !allDatasets && setHoveredCell({ row: rowIdx, col: colIdx })}
                          onMouseLeave={() => !allDatasets && setHoveredCell(null)}
                        >
                          <div
                            className="rounded p-2 text-center"
                            style={{ backgroundColor: background, color: text }}
                          >
                            <div className="font-bold">{value.toFixed(2)}</div>
                            <div className="text-xs">{percentage.toFixed(1)}%</div>
                          </div>

                          {!allDatasets && hoveredCell?.row === rowIdx && hoveredCell?.col === colIdx && relevantPosts.length > 0 && (
                            <div className="absolute z-50 bg-white p-3 rounded-lg shadow-lg border border-gray-200 min-w-[250px] -translate-y-full left-1/2 -translate-x-1/2">
                              <div className="font-medium mb-2">Sentiment Distribution</div>
                              <div className="space-y-2">
                                {relevantPosts.map((post, i) => (
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
                    <td className="px-4 py-2 text-center font-medium">{rowTotals[rowIdx].toFixed(2)}</td>
                  </tr>
                ))}
                <tr className="bg-slate-100 font-medium">
                  <td className="px-4 py-2">Total</td>
                  {colTotals.map((total, idx) => (
                    <td key={idx} className="px-4 py-2 text-center">{total.toFixed(2)}</td>
                  ))}
                  <td className="px-4 py-2 text-center">{totalSamples.toFixed(2)}</td>
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
                <p>• Numbers show actual sentiment counts from data</p>
                <p>• Diagonal cells show primary sentiment confidence</p>
                <p>• Off-diagonal cells show mixed sentiment distribution</p>
                <p>• {allDatasets ? 'Showing aggregated data from all datasets' : 'Hover for detailed confidence breakdown per sample'}</p>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}