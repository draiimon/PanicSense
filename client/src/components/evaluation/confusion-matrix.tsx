import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { useQuery } from '@tanstack/react-query';
import { getSentimentPostsByFileId } from '@/lib/api';
import { getSentimentColor } from '@/lib/colors';
import { Badge } from '@/components/ui/badge';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

interface ConfusionMatrixProps {
  fileId?: number;
  confusionMatrix?: number[][];
  labels?: string[];
  title?: string;
  description?: string;
  allDatasets?: boolean;
}

const defaultLabels = ['Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral'];

export function ConfusionMatrix({
  fileId,
  confusionMatrix: initialMatrix,
  labels = defaultLabels,
  title = 'Confusion Matrix',
  description = 'Real sentiment distribution with confidence scores',
  allDatasets = false
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

  // Fetch sentiment posts if fileId is provided and not in allDatasets mode
  const { data: sentimentPosts, isLoading } = useQuery({
    queryKey: ['/api/sentiment-posts/file', fileId],
    queryFn: () => getSentimentPostsByFileId(fileId as number),
    enabled: !!fileId && !initialMatrix && !allDatasets
  });

  // Process sentiment data and build confusion matrix
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
            id: post.id.toString(),
            mainSentiment,
            confidence,
            mixedSentiments: postSentiments
          });
        }
      });
    }

    // Calculate metrics for visualization
    const metrics = labels.map((_, idx) => {
      const truePositive = newMatrix[idx][idx];
      const rowSum = newMatrix[idx].reduce((sum, val) => sum + val, 0);
      const colSum = newMatrix.reduce((sum, row) => sum + row[idx], 0);
      const totalSum = newMatrix.reduce((sum, row) => sum + row.reduce((s, v) => s + v, 0), 0);

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

    setMatrix(newMatrix);
    setSentimentData(newSentimentData);
    setMetricsData(metrics);
    setIsMatrixCalculated(true);
  }, [sentimentPosts, labels, initialMatrix, isLoading, allDatasets]);

  if (isLoading || !isMatrixCalculated) {
    return (
      <Card className="bg-white rounded-lg shadow-md">
        <CardContent className="p-6 text-center">
          <div className="flex items-center justify-center h-40">
            <svg className="animate-spin h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p className="ml-3 text-slate-500">Calculating metrics...</p>
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
          {/* Enhanced Metrics Visualization */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Performance Metrics Line Chart */}
            <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">Performance Trends</h3>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metricsData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="sentiment" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="precision" stroke="#22c55e" name="Precision" />
                    <Line type="monotone" dataKey="recall" stroke="#8b5cf6" name="Recall" />
                    <Line type="monotone" dataKey="f1Score" stroke="#f97316" name="F1 Score" />
                    <Line type="monotone" dataKey="accuracy" stroke="#3b82f6" name="Accuracy" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Radar Chart for Balanced View */}
            <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
              <h3 className="text-lg font-semibold mb-4">Metric Balance</h3>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="80%" data={metricsData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="sentiment" />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} />
                    <Radar name="Precision" dataKey="precision" stroke="#22c55e" fill="#22c55e" fillOpacity={0.6} />
                    <Radar name="Recall" dataKey="recall" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
                    <Legend />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Confusion Matrix Table */}
          <div className="overflow-x-auto mt-6">
            <table className="w-full">
              <thead>
                <tr className="bg-slate-50">
                  <th className="px-4 py-2 text-left">True Sentiment</th>
                  {labels.map((label, idx) => (
                    <th key={idx} className="px-4 py-2 text-center">{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.map((row, rowIdx) => (
                  <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                    <td className="px-4 py-2 font-medium">{labels[rowIdx]}</td>
                    {row.map((value, colIdx) => {
                      const color = getSentimentColor(labels[colIdx]);
                      const percentage = (value / row.reduce((sum, val) => sum + val, 0) * 100) || 0;

                      return (
                        <td
                          key={colIdx}
                          className="px-4 py-2 relative"
                          onMouseEnter={() => setHoveredCell({ row: rowIdx, col: colIdx })}
                          onMouseLeave={() => setHoveredCell(null)}
                        >
                          <div
                            className="rounded p-2 text-center transition-all duration-200"
                            style={{
                              backgroundColor: `${color}${Math.floor(percentage).toString(16).padStart(2, '0')}`,
                              color: percentage > 40 ? '#ffffff' : '#333333'
                            }}
                          >
                            <div className="font-bold">{value.toFixed(2)}</div>
                            <div className="text-xs">{percentage.toFixed(1)}%</div>
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Legend */}
          <div className="bg-slate-50 p-4 rounded-lg mt-4">
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
        </div>
      </CardContent>
    </Card>
  );
}