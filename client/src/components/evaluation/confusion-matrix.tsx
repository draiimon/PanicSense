import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
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
  const [metricsData, setMetricsData] = useState<any[]>([]);

  const { data: sentimentPosts, isLoading } = useQuery({
    queryKey: ['/api/sentiment-posts/file', fileId],
    queryFn: () => getSentimentPostsByFileId(fileId as number),
    enabled: !!fileId && !initialMatrix && !allDatasets
  });

  useEffect(() => {
    if ((isLoading || !sentimentPosts) && !initialMatrix) return;

    // Initialize confusion matrix
    let newMatrix: number[][] = initialMatrix 
      ? initialMatrix.map(row => [...row])
      : Array(labels.length).fill(0).map(() => Array(labels.length).fill(0));

    // If we have sentiment posts, build the matrix
    if (!initialMatrix && sentimentPosts && sentimentPosts.length > 0) {
      sentimentPosts.forEach(post => {
        const mainSentiment = post.sentiment;
        const mainIdx = labels.findIndex(label => label === mainSentiment);
        if (mainIdx === -1) return;
        newMatrix[mainIdx][mainIdx]++;
      });
    }

    // Calculate metrics directly from confusion matrix
    const metrics = labels.map((label, idx) => {
      // True positives are diagonal elements
      const truePositive = newMatrix[idx][idx];

      // Row sum represents actual instances of this class
      const rowSum = newMatrix[idx].reduce((sum, val) => sum + val, 0);

      // Column sum represents predicted instances of this class
      const colSum = newMatrix.reduce((sum, row) => sum + row[idx], 0);

      // Calculate total samples
      const totalSamples = newMatrix.reduce((sum, row) => 
        sum + row.reduce((s, v) => s + v, 0), 0
      );

      // Calculate metrics
      const precision = colSum === 0 ? 0 : (truePositive / colSum) * 100;
      const recall = rowSum === 0 ? 0 : (truePositive / rowSum) * 100;
      const f1Score = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
      const accuracy = totalSamples === 0 ? 0 : 
        ((newMatrix[idx][idx] + // True Positives
          newMatrix.reduce((sum, row, i) => // True Negatives
            sum + (i !== idx ? row.reduce((s, v, j) => s + (j !== idx ? v : 0), 0) : 0), 0)
        ) / totalSamples) * 100;

      return {
        sentiment: label,
        precision: Math.round(precision * 10) / 10,
        recall: Math.round(recall * 10) / 10,
        f1Score: Math.round(f1Score * 10) / 10,
        accuracy: Math.round(accuracy * 10) / 10
      };
    });

    setMatrix(newMatrix);
    setMetricsData(metrics);
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

  if (isLoading || !isMatrixCalculated) {
    return (
      <Card className="bg-white rounded-lg shadow">
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
    <div className="space-y-8">
      {/* Performance Metrics Bar Chart */}
      <Card className="bg-white rounded-lg shadow-md">
        <CardHeader className="px-6 py-4 border-b border-gray-200">
          <CardTitle className="text-lg font-semibold text-slate-800">Performance Metrics Bar Chart</CardTitle>
          <CardDescription className="text-sm text-slate-600">
            Comparative view of key metrics as bar charts for each sentiment category
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="sentiment" />
                <YAxis domain={[0, 100]} />
                <Tooltip 
                  formatter={(value: number) => [`${value.toFixed(1)}%`]}
                />
                <Legend />
                <Bar dataKey="precision" fill="#22c55e" name="Precision" />
                <Bar dataKey="recall" fill="#8b5cf6" name="Recall" />
                <Bar dataKey="f1Score" fill="#f97316" name="F1 Score" />
                <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Confusion Matrix */}
      <Card className="bg-white rounded-lg shadow-md">
        <CardHeader className="px-6 py-4 border-b border-gray-200">
          <CardTitle className="text-lg font-semibold text-slate-800">Sentiment Confusion Matrix</CardTitle>
          <CardDescription className="text-sm text-slate-600">
            Distribution of predicted vs actual sentiments
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          <div className="overflow-x-auto">
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
                      const { background, text } = getCellColor(rowIdx, colIdx, value);
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
                            style={{ backgroundColor: background, color: text }}
                          >
                            <div className="text-lg font-bold">{percentage.toFixed(1)}%</div>
                            <div className="text-xs opacity-75">({value})</div>
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Metrics Understanding */}
      <div className="grid grid-cols-1 gap-6">
        <Card className="bg-white rounded-lg shadow-md">
          <CardHeader className="px-6 py-4 border-b border-gray-200">
            <CardTitle className="text-lg font-semibold text-slate-800">Understanding the Metrics</CardTitle>
            <CardDescription className="text-sm text-slate-600">
              How to interpret the confusion matrix and performance metrics
            </CardDescription>
          </CardHeader>
          <CardContent className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="font-medium text-slate-800">Performance Metrics</h3>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="w-3 h-3 rounded-full bg-[#22c55e] mt-1.5" />
                    <div>
                      <p className="font-medium text-slate-800">Precision</p>
                      <p className="text-sm text-slate-600">Of all predicted instances for each sentiment, what percentage was correct</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-3 h-3 rounded-full bg-[#8b5cf6] mt-1.5" />
                    <div>
                      <p className="font-medium text-slate-800">Recall</p>
                      <p className="text-sm text-slate-600">Of all actual instances of each sentiment, what percentage was correctly identified</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-3 h-3 rounded-full bg-[#f97316] mt-1.5" />
                    <div>
                      <p className="font-medium text-slate-800">F1 Score</p>
                      <p className="text-sm text-slate-600">Balanced measure between precision and recall (higher is better)</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-3 h-3 rounded-full bg-[#3b82f6] mt-1.5" />
                    <div>
                      <p className="font-medium text-slate-800">Accuracy</p>
                      <p className="text-sm text-slate-600">Overall correct predictions for each sentiment category</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="font-medium text-slate-800">Confusion Matrix</h3>
                <div className="space-y-2">
                  <p className="text-sm text-slate-600">• Each cell shows the percentage and count of predictions</p>
                  <p className="text-sm text-slate-600">• The rows represent the true (actual) sentiment</p>
                  <p className="text-sm text-slate-600">• The columns show what the model predicted</p>
                  <p className="text-sm text-slate-600">• Diagonal cells (darker colors) show correct predictions</p>
                  <p className="text-sm text-slate-600">• Higher percentages in diagonal cells indicate better performance</p>
                </div>
                <div className="mt-4">
                  <h4 className="font-medium text-slate-800 mb-2">Sentiment Categories</h4>
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
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}