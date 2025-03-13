import { useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { chartColors } from '@/lib/colors';
import Chart from 'chart.js/auto';

interface MetricsData {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: number[][];
}

interface MetricsDisplayProps {
  data?: MetricsData;
  title?: string;
  description?: string;
}

export function MetricsDisplay({ 
  data, 
  title = 'Evaluation Metrics',
  description = 'Model performance metrics'
}: MetricsDisplayProps) {
  const metricsChartRef = useRef<HTMLCanvasElement>(null);
  const confusionMatrixChartRef = useRef<HTMLCanvasElement>(null);
  const metricsChartInstance = useRef<Chart | null>(null);
  const confusionMatrixChartInstance = useRef<Chart | null>(null);

  const sentimentLabels = ['Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral'];

  useEffect(() => {
    if (!data) return;

    // Metrics Chart
    if (metricsChartRef.current) {
      if (metricsChartInstance.current) {
        metricsChartInstance.current.destroy();
      }

      const ctx = metricsChartRef.current.getContext('2d');
      if (!ctx) return;

      metricsChartInstance.current = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
          datasets: [{
            data: [data.accuracy, data.precision, data.recall, data.f1Score],
            backgroundColor: ['#4299e1', '#48bb78', '#ed8936', '#9f7aea'],
            borderWidth: 0
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              max: 1,
              ticks: {
                callback: function(value) {
                  return (value as number * 100).toFixed(0) + '%';
                }
              }
            }
          },
          plugins: {
            legend: {
              display: false
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  return (context.raw as number * 100).toFixed(1) + '%';
                }
              }
            }
          }
        }
      });
    }

    // Confusion Matrix Chart
    if (confusionMatrixChartRef.current && data.confusionMatrix) {
      if (confusionMatrixChartInstance.current) {
        confusionMatrixChartInstance.current.destroy();
      }

      const ctx = confusionMatrixChartRef.current.getContext('2d');
      if (!ctx) return;

      // Flatten confusion matrix for heatmap
      const matrixData = [];
      const matrixLabels = [];
      
      for (let i = 0; i < data.confusionMatrix.length; i++) {
        for (let j = 0; j < data.confusionMatrix[i].length; j++) {
          matrixData.push({
            x: j,
            y: i,
            v: data.confusionMatrix[i][j]
          });
        }
      }

      confusionMatrixChartInstance.current = new Chart(ctx, {
        type: 'scatter',
        data: {
          datasets: [{
            label: 'Confusion Matrix',
            data: matrixData,
            backgroundColor: (context) => {
              const value = context.raw?.v;
              // Create a color gradient based on the value
              const alpha = Math.min(1, Math.max(0.1, value / Math.max(...data.confusionMatrix.flat())));
              return `rgba(66, 153, 225, ${alpha})`;
            },
            borderColor: 'rgba(0, 0, 0, 0.1)',
            borderWidth: 1,
            pointRadius: 0,
            pointHoverRadius: 0,
            pointHitRadius: 0
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              type: 'linear',
              position: 'bottom',
              min: -0.5,
              max: data.confusionMatrix[0].length - 0.5,
              ticks: {
                callback: function(value) {
                  return sentimentLabels[value as number] || '';
                },
                stepSize: 1,
                font: {
                  weight: 'bold'
                },
                color: '#4A5568'  // Darker color for better readability
              },
              title: {
                display: true,
                text: 'Predicted Emotion',
                font: {
                  weight: 'bold',
                  size: 14
                }
              }
            },
            y: {
              type: 'linear',
              min: -0.5,
              max: data.confusionMatrix.length - 0.5,
              ticks: {
                callback: function(value) {
                  return sentimentLabels[value as number] || '';
                },
                stepSize: 1,
                reverse: true
              },
              title: {
                display: true,
                text: 'True'
              }
            }
          },
          plugins: {
            legend: {
              display: false
            },
            tooltip: {
              callbacks: {
                title: function() {
                  return '';
                },
                label: function(context) {
                  const dataPoint = context.raw as { x: number, y: number, v: number };
                  const trueLabel = sentimentLabels[dataPoint.y];
                  const predictedLabel = sentimentLabels[dataPoint.x];
                  return [
                    `True: ${trueLabel}`,
                    `Predicted: ${predictedLabel}`,
                    `Count: ${dataPoint.v}`
                  ];
                }
              }
            }
          }
        }
      });
    }

    return () => {
      if (metricsChartInstance.current) {
        metricsChartInstance.current.destroy();
      }
      if (confusionMatrixChartInstance.current) {
        confusionMatrixChartInstance.current.destroy();
      }
    };
  }, [data]);

  if (!data) {
    return (
      <Card className="bg-white rounded-lg shadow">
        <CardHeader className="p-5 border-b border-gray-200">
          <CardTitle className="text-lg font-medium text-slate-800">{title}</CardTitle>
          <CardDescription className="text-sm text-slate-500">{description}</CardDescription>
        </CardHeader>
        <CardContent className="p-5 text-center py-12">
          <p className="text-slate-500">No evaluation metrics available</p>
          <p className="text-sm text-slate-400 mt-2">Upload a CSV file to generate metrics</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Metrics Chart */}
      <Card className="bg-white rounded-lg shadow">
        <CardHeader className="p-5 border-b border-gray-200">
          <CardTitle className="text-lg font-medium text-slate-800">Performance Metrics</CardTitle>
          <CardDescription className="text-sm text-slate-500">
            Accuracy, Precision, Recall, F1 Score
          </CardDescription>
        </CardHeader>
        <CardContent className="p-5">
          <div className="h-60">
            <canvas ref={metricsChartRef} />
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div className="text-center p-3 bg-slate-50 rounded-lg">
              <p className="text-sm text-slate-500">Accuracy</p>
              <p className="text-xl font-bold text-slate-800">
                {(data.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-center p-3 bg-slate-50 rounded-lg">
              <p className="text-sm text-slate-500">Precision</p>
              <p className="text-xl font-bold text-slate-800">
                {(data.precision * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-center p-3 bg-slate-50 rounded-lg">
              <p className="text-sm text-slate-500">Recall</p>
              <p className="text-xl font-bold text-slate-800">
                {(data.recall * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-center p-3 bg-slate-50 rounded-lg">
              <p className="text-sm text-slate-500">F1 Score</p>
              <p className="text-xl font-bold text-slate-800">
                {(data.f1Score * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Confusion Matrix */}
      <Card className="bg-white rounded-lg shadow">
        <CardHeader className="p-5 border-b border-gray-200">
          <CardTitle className="text-lg font-medium text-slate-800">Confusion Matrix</CardTitle>
          <CardDescription className="text-sm text-slate-500">
            True vs Predicted Sentiments
          </CardDescription>
        </CardHeader>
        <CardContent className="p-5">
          <div className="h-80">
            <canvas ref={confusionMatrixChartRef} />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
