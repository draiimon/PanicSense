import { useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { chartColors } from '@/lib/colors';
import Chart from 'chart.js/auto';

interface MetricsData {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
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
  const metricsChartInstance = useRef<Chart | null>(null);

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
  }, [data]);

  if (!data) {
    return (
      <Card className="bg-white rounded-lg shadow">
        <CardHeader className="p-5 border-b border-gray-200">
          <CardTitle className="text-lg font-medium text-slate-800">{title}</CardTitle>
          <CardDescription className="text-sm text-slate-500">{description}</CardDescription>
        </CardHeader>
        <CardContent className="p-5">
          <div className="flex items-center justify-center h-48">
            <p className="text-slate-500">No metrics data available</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Performance Metrics */}
      <Card className="bg-white rounded-lg shadow">
        <CardHeader className="p-5 border-b border-gray-200">
          <CardTitle className="text-lg font-medium text-slate-800">{title}</CardTitle>
          <CardDescription className="text-sm text-slate-500">{description}</CardDescription>
        </CardHeader>
        <CardContent className="p-5">
          <div className="h-64 mb-6">
            <canvas ref={metricsChartRef} />
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
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
    </div>
  );
}