import React, { useRef, useEffect, useMemo } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import Chart from 'chart.js/auto';
import { sentimentColors } from '@/lib/colors';
import { motion } from 'framer-motion';

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

export function OptimizedSentimentChart({ 
  data,
  type = 'doughnut',
  height = '300px'
}: SentimentChartProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);

  // Memoize colors to prevent recalculation on every render
  const colors = useMemo(() => {
    return data.labels.map(label => {
      const sanitizedKey = label.toLowerCase().replace(/\//g, '_');
      const colorKey = sanitizedKey as keyof typeof sentimentColors;
      const color = sentimentColors[colorKey] || '#6b7280';
      return color;
    });
  }, [data.labels]);

  // Memoize chart options to prevent recreation on every render
  const chartOptions = useMemo(() => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 800, // slower animations to reduce flickering
        easing: 'easeOutQuart' as const
      },
      plugins: {
        legend: {
          display: true,
          position: 'bottom' as const,
          labels: {
            font: {
              size: 12,
            },
            padding: 15,
            usePointStyle: true,
          },
        },
        tooltip: {
          enabled: true,
          mode: 'index' as const,
          intersect: false,
          backgroundColor: 'rgba(17, 24, 39, 0.8)',
          titleFont: {
            size: 13,
          },
          bodyFont: {
            size: 12,
          },
          padding: 10,
          cornerRadius: 4,
          displayColors: true,
        },
      },
      scales: type !== 'doughnut' ? {
        x: {
          display: true,
          grid: {
            display: false,
          },
          ticks: {
            font: {
              size: 11,
            },
          },
        },
        y: {
          display: true,
          beginAtZero: true,
          grid: {
            color: 'rgba(0, 0, 0, 0.05)',
          },
          ticks: {
            font: {
              size: 11,
            },
            precision: 0,
          },
        },
      } : undefined,
    };
  }, [type]);

  // Update chart instead of recreating when possible
  useEffect(() => {
    if (!chartRef.current) return;
    
    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;
    
    // If chart exists, update data instead of destroying
    if (chartInstance.current) {
      chartInstance.current.data.labels = data.labels;
      chartInstance.current.data.datasets[0].data = data.values;
      chartInstance.current.data.datasets[0].backgroundColor = colors;
      
      if (type === 'line') {
        chartInstance.current.data.datasets[0].borderColor = colors;
      } else {
        chartInstance.current.data.datasets[0].borderColor = Array(data.labels.length).fill('rgba(255, 255, 255, 0.8)');
      }
      
      chartInstance.current.update('none'); // update without animation first time
      return;
    }

    // Create new chart if it doesn't exist
    chartInstance.current = new Chart(ctx, {
      type,
      data: {
        labels: data.labels,
        datasets: [
          {
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
          },
        ],
      },
      options: chartOptions,
    });

    // Cleanup function
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
        chartInstance.current = null;
      }
    };
  }, [data, colors, type, chartOptions]);

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