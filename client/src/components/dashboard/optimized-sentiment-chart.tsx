import React, { useRef, useEffect, useMemo } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import Chart from 'chart.js/auto';
import { sentimentColors } from '@/lib/colors';
import { motion } from 'framer-motion';
import { Skeleton } from "@/components/ui/skeleton";

interface SentimentChartProps {
  data: {
    labels: string[];
    values: number[];
    title?: string;
    description?: string;
  };
  type?: 'doughnut' | 'bar' | 'line';
  height?: string;
  isLoading?: boolean;
}

export function OptimizedSentimentChart({ 
  data,
  type = 'doughnut',
  height = '300px',
  isLoading = false
}: SentimentChartProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);

  // Memoize colors to prevent recalculation
  const colors = useMemo(() => {
    return data.labels.map(label => {
      switch (label) {
        case 'Panic': return '#ef4444';
        case 'Fear/Anxiety': return '#f97316';
        case 'Disbelief': return '#8b5cf6';
        case 'Resilience': return '#10b981';
        case 'Neutral':
        default: return '#6b7280';
      }
    });
  }, [data.labels]);

  // Memoize chart options
  const chartOptions = useMemo(() => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false, // Disable animations to prevent flickering
      plugins: {
        legend: {
          display: true,
          position: 'bottom' as const,
          labels: {
            font: { size: 12 },
            padding: 15,
            usePointStyle: true,
          },
        },
        tooltip: {
          enabled: true,
          mode: 'index' as const,
          intersect: false,
          backgroundColor: 'rgba(17, 24, 39, 0.8)',
          titleFont: { size: 13 },
          bodyFont: { size: 12 },
          padding: 10,
          cornerRadius: 4,
          displayColors: true,
        },
      },
      scales: type !== 'doughnut' ? {
        x: {
          display: true,
          grid: { display: false },
          ticks: { font: { size: 11 } },
        },
        y: {
          display: true,
          beginAtZero: true,
          grid: { color: 'rgba(0, 0, 0, 0.05)' },
          ticks: {
            font: { size: 11 },
            precision: 0,
          },
        },
      } : undefined,
    };
  }, [type]);

  // Update or create chart
  useEffect(() => {
    if (isLoading || !chartRef.current) return;

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;

    const updateChart = () => {
      if (chartInstance.current) {
        // Update existing chart
        chartInstance.current.data.labels = data.labels;
        chartInstance.current.data.datasets[0].data = data.values;
        chartInstance.current.data.datasets[0].backgroundColor = colors;
        chartInstance.current.options = chartOptions;
        chartInstance.current.update('none'); // Update without animation
      } else {
        // Create new chart
        chartInstance.current = new Chart(ctx, {
          type,
          data: {
            labels: data.labels,
            datasets: [{
              label: 'Sentiment Distribution',
              data: data.values,
              backgroundColor: colors,
              borderColor: type === 'line' ? colors : Array(data.labels.length).fill('rgba(255, 255, 255, 0.8)'),
              borderWidth: 2,
              tension: 0.3,
              fill: type === 'line',
              pointBackgroundColor: type === 'line' ? '#fff' : undefined,
              pointBorderColor: type === 'line' ? colors : undefined,
              pointRadius: type === 'line' ? 4 : undefined,
              pointHoverRadius: type === 'line' ? 6 : undefined,
            }],
          },
          options: chartOptions,
        });
      }
    };

    // Delay chart update to prevent flickering
    const timeoutId = setTimeout(updateChart, 100);

    return () => {
      clearTimeout(timeoutId);
      if (chartInstance.current) {
        chartInstance.current.destroy();
        chartInstance.current = null;
      }
    };
  }, [data, colors, type, chartOptions, isLoading]);

  if (isLoading) {
    return (
      <div style={{ height }} className="flex items-center justify-center">
        <div className="space-y-4 w-full">
          <Skeleton className="h-[200px] w-full bg-gray-200/50" />
          <div className="flex justify-center space-x-4">
            {[1, 2, 3, 4, 5].map((i) => (
              <Skeleton key={i} className="h-4 w-20 bg-gray-200/50" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <motion.div 
      style={{ height }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="chart-container"
    >
      <canvas ref={chartRef}></canvas>
    </motion.div>
  );
}