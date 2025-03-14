import { useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { chartColors } from '@/lib/colors';
import Chart from 'chart.js/auto';

interface SentimentChartProps {
  data: {
    labels: string[];
    values: number[];
    title?: string;
    description?: string;
  };
  type?: 'doughnut' | 'bar' | 'line';
  height?: string;
}

export function SentimentChart({ 
  data, 
  type = 'doughnut',
  height = 'h-80'
}: SentimentChartProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);

  useEffect(() => {
    if (chartRef.current) {
      // Destroy previous chart if it exists
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }

      const ctx = chartRef.current.getContext('2d');
      if (!ctx) return;

      // Chart configuration based on type
      let chartConfig: any = {
        type,
        data: {
          labels: data.labels,
          datasets: [{
            data: data.values,
            backgroundColor: chartColors.slice(0, data.labels.length),
            borderWidth: 0
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
        }
      };

      // Type-specific configurations
      if (type === 'doughnut') {
        chartConfig.options.cutout = '70%';
        chartConfig.options.plugins = {
          legend: {
            position: 'bottom'
          }
        };
      } else if (type === 'bar' || type === 'line') {
        chartConfig.options.scales = {
          y: {
            beginAtZero: true
          }
        };
        
        if (type === 'line') {
          chartConfig.data.datasets[0].tension = 0.4;
          chartConfig.data.datasets[0].fill = true;
          chartConfig.data.datasets = data.labels.map((label, index) => ({
            label,
            data: [data.values[index]],
            borderColor: chartColors[index % chartColors.length],
            backgroundColor: `${chartColors[index % chartColors.length]}33`,
            tension: 0.4,
            fill: true
          }));
        }
      }

      chartInstance.current = new Chart(ctx, chartConfig);
    }

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [data, type]);

  return (
    <Card className="bg-white rounded-lg shadow">
      <CardHeader className="p-5 border-b border-gray-200">
        <CardTitle className="text-lg font-medium text-slate-800">
          {data.title || 'Sentiment Distribution'}
        </CardTitle>
        <CardDescription className="text-sm text-slate-500">
          {data.description || 'Across all active disasters'}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-5">
        <div className={height}>
          <canvas ref={chartRef} />
        </div>
      </CardContent>
    </Card>
  );
}
