import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { chartColors } from '@/lib/colors';
import Chart from 'chart.js/auto';
import 'chartjs-adapter-date-fns';

interface TimelineData {
  labels: string[]; // dates
  datasets: {
    label: string;
    data: number[];
  }[];
}

interface SentimentTimelineProps {
  data: TimelineData;
  title?: string;
  description?: string;
}

export function SentimentTimeline({ 
  data, 
  title = 'Sentiment Evolution',
  description = 'Last 7 days'
}: SentimentTimelineProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);
  const [timeRange, setTimeRange] = useState<'day' | 'week' | 'month'>('day');

  useEffect(() => {
    if (chartRef.current) {
      // Destroy previous chart if it exists
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }

      const ctx = chartRef.current.getContext('2d');
      if (!ctx) return;

      // Format datasets with colors
      const formattedDatasets = data.datasets.map((dataset, index) => ({
        ...dataset,
        borderColor: chartColors[index % chartColors.length],
        backgroundColor: `${chartColors[index % chartColors.length]}20`,
        fill: true,
        tension: 0.4
      }));

      // Create chart
      chartInstance.current = new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.labels,
          datasets: formattedDatasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              max: 50,
              title: {
                display: true,
                text: 'Sentiment Percentage'
              }
            },
            x: {
              title: {
                display: true,
                text: 'Date'
              }
            }
          },
          plugins: {
            legend: {
              position: 'bottom'
            }
          }
        }
      });
    }

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [data, timeRange]);

  return (
    <Card className="bg-white rounded-lg shadow">
      <CardHeader className="p-5 border-b border-gray-200 flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle className="text-lg font-medium text-slate-800">{title}</CardTitle>
          <CardDescription className="text-sm text-slate-500">{description}</CardDescription>
        </div>
        <div className="mt-3 sm:mt-0 flex space-x-3">
          <Button 
            size="sm" 
            variant={timeRange === 'day' ? 'default' : 'outline'}
            onClick={() => setTimeRange('day')}
            className="text-xs"
          >
            Day
          </Button>
          <Button 
            size="sm" 
            variant={timeRange === 'week' ? 'default' : 'outline'}
            onClick={() => setTimeRange('week')}
            className="text-xs"
          >
            Week
          </Button>
          <Button 
            size="sm" 
            variant={timeRange === 'month' ? 'default' : 'outline'}
            onClick={() => setTimeRange('month')}
            className="text-xs"
          >
            Month
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-5">
        <div className="h-80">
          <canvas ref={chartRef} />
        </div>
      </CardContent>
    </Card>
  );
}
