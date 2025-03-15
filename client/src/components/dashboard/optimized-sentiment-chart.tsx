import React, { useRef, useEffect, useMemo } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import Chart from 'chart.js/auto';
import { sentimentColors } from '@/lib/colors';
import { motion, AnimatePresence } from 'framer-motion';
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

  // Prevent updates during loading
  const processedData = useMemo(() => {
    if (isLoading) {
      return {
        labels: [],
        values: []
      };
    }
    return data;
  }, [data, isLoading]);

  // Memoize colors to prevent recalculation
  const colors = useMemo(() => {
    return processedData.labels.map(label => {
      switch (label) {
        case 'Panic': return '#ef4444';
        case 'Fear/Anxiety': return '#f97316';
        case 'Disbelief': return '#8b5cf6';
        case 'Resilience': return '#10b981';
        case 'Neutral':
        default: return '#6b7280';
      }
    });
  }, [processedData.labels]);

  // Memoize chart options
  const chartOptions = useMemo(() => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: isLoading ? 0 : 800, // Disable animations during loading
        easing: 'easeOutQuart' as const
      },
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
          enabled: !isLoading, // Disable tooltips during loading
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
  }, [type, isLoading]);

  // Update or create chart
  useEffect(() => {
    if (!chartRef.current) return;

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;

    // During loading, either destroy existing chart or do nothing
    if (isLoading) {
      if (chartInstance.current) {
        chartInstance.current.destroy();
        chartInstance.current = null;
      }
      return;
    }

    // Create chart configuration
    const chartConfig = {
      type,
      data: {
        labels: processedData.labels,
        datasets: [{
          label: 'Sentiment Distribution',
          data: processedData.values,
          backgroundColor: colors,
          borderColor: type === 'line' ? colors : Array(processedData.labels.length).fill('rgba(255, 255, 255, 0.8)'),
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
    };

    // If chart exists and not loading, update it
    if (chartInstance.current) {
      chartInstance.current.data = chartConfig.data;
      chartInstance.current.options = chartConfig.options;
      chartInstance.current.update('none');
      return;
    }

    // Create new chart if none exists
    chartInstance.current = new Chart(ctx, chartConfig);

    // Cleanup function
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
        chartInstance.current = null;
      }
    };
  }, [processedData, colors, type, chartOptions, isLoading]);

  // Loading state UI
  if (isLoading) {
    return (
      <div style={{ height }} className="flex items-center justify-center">
        <div className="space-y-4 w-full">
          <Skeleton className="h-[200px] w-full bg-gray-200 animate-pulse" />
          <div className="flex justify-center space-x-4">
            {[1, 2, 3, 4, 5].map((i) => (
              <Skeleton key={i} className="h-4 w-20 bg-gray-200 animate-pulse" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Render chart with animations
  return (
    <AnimatePresence mode="wait">
      <motion.div 
        style={{ height }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.5 }}
        className="chart-container"
      >
        <canvas ref={chartRef}></canvas>
      </motion.div>
    </AnimatePresence>
  );
}