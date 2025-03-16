import React, { useRef, useEffect, useMemo } from 'react';
import Chart from 'chart.js/auto';
import { sentimentColors } from '@/lib/colors';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Loader2, 
  PieChart, 
  BarChart, 
  LineChart, 
  BarChart3, 
  Info
} from "lucide-react";

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
  
  // Check if the data is empty (all values are 0)
  const isEmpty = useMemo(() => {
    return !isLoading && data.values.every(val => val === 0);
  }, [data.values, isLoading]);

  // Prevent updates during loading
  const processedData = useMemo(() => {
    if (isLoading || isEmpty) {
      return {
        labels: [],
        values: []
      };
    }
    return data;
  }, [data, isLoading, isEmpty]);

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
        duration: isLoading ? 0 : 800,
        easing: 'easeOutQuart' as const
      },
      plugins: {
        legend: {
          display: true,
          position: 'bottom' as const,
          labels: {
            font: { 
              family: "'Inter', sans-serif",
              size: 12,
              weight: 500
            },
            padding: 20,
            usePointStyle: true,
            pointStyleWidth: 10,
            boxWidth: 10,
            boxHeight: 10,
          },
        },
        tooltip: {
          enabled: !isLoading,
          mode: 'index' as const,
          intersect: false,
          backgroundColor: 'rgba(30, 41, 59, 0.8)',
          titleFont: { 
            family: "'Inter', sans-serif",
            size: 13,
            weight: 600
          },
          bodyFont: { 
            family: "'Inter', sans-serif",
            size: 12 
          },
          padding: 12,
          cornerRadius: 6,
          displayColors: true,
          boxPadding: 6,
          callbacks: {
            label: function(context: any) {
              // Add number and percentage 
              const total = context.dataset.data.reduce((a: number, b: number) => a + b, 0);
              const value = context.raw;
              const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
              return `${context.label}: ${value} (${percentage}%)`;
            }
          }
        },
      },
      scales: type !== 'doughnut' ? {
        x: {
          display: true,
          grid: { 
            display: false,
            drawBorder: false,
          },
          ticks: { 
            font: { 
              family: "'Inter', sans-serif",
              size: 11,
              weight: 500
            },
            color: '#64748b',
            padding: 8,
          },
        },
        y: {
          display: true,
          beginAtZero: true,
          grid: { 
            color: 'rgba(203, 213, 225, 0.4)',
            drawBorder: false,
          },
          border: {
            display: false,
          },
          ticks: {
            font: { 
              family: "'Inter', sans-serif",
              size: 11,
              weight: 500
            },
            color: '#64748b',
            padding: 8,
            precision: 0,
          },
        },
      } : undefined,
      cutout: type === 'doughnut' ? '65%' : undefined,
      radius: type === 'doughnut' ? '90%' : undefined,
    };
  }, [type, isLoading]);

  // Update or create chart
  useEffect(() => {
    if (!chartRef.current || isLoading || isEmpty) return;

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;

    // Create chart configuration
    const chartConfig = {
      type,
      data: {
        labels: processedData.labels,
        datasets: [{
          label: 'Sentiment Distribution',
          data: processedData.values,
          backgroundColor: colors,
          borderColor: type === 'line' ? colors : Array(processedData.labels.length).fill('#ffffff'),
          borderWidth: type === 'doughnut' ? 3 : 2,
          borderRadius: type === 'bar' ? 6 : undefined,
          hoverBorderWidth: type === 'doughnut' ? 4 : 2,
          hoverOffset: type === 'doughnut' ? 10 : undefined,
          tension: 0.3,
          fill: type === 'line',
          pointBackgroundColor: type === 'line' ? '#fff' : undefined,
          pointBorderColor: type === 'line' ? colors : undefined,
          pointRadius: type === 'line' ? 5 : undefined,
          pointHoverRadius: type === 'line' ? 7 : undefined,
          pointBorderWidth: type === 'line' ? 2 : undefined,
          pointHoverBorderWidth: type === 'line' ? 3 : undefined,
          shadowOffsetX: 0,
          shadowOffsetY: 6,
          shadowBlur: 10,
          shadowColor: 'rgba(0, 0, 0, 0.2)',
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
  }, [processedData, colors, type, chartOptions, isLoading, isEmpty]);

  // Get the appropriate chart icon based on type
  const getChartIcon = () => {
    switch (type) {
      case 'bar': return <BarChart3 className="h-8 w-8 text-blue-400" />;
      case 'line': return <LineChart className="h-8 w-8 text-blue-400" />;
      case 'doughnut': 
      default: return <PieChart className="h-8 w-8 text-blue-400" />;
    }
  };

  return (
    <div style={{ height }} className="relative">
      {/* Loading State */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="absolute inset-0 bg-white/50 backdrop-blur-sm"></div>
          <div className="bg-white rounded-xl p-5 shadow-sm flex items-center gap-4 z-10">
            <div className="relative h-10 w-10 flex-shrink-0">
              <Loader2 className="h-10 w-10 absolute inset-0 text-blue-600/20" />
              <Loader2 className="h-10 w-10 absolute inset-0 text-blue-600 animate-spin" />
            </div>
            <div>
              <p className="font-medium text-slate-800">Processing sentiment data...</p>
              <p className="text-xs text-slate-500">Analyzing emotional responses</p>
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {isEmpty && !isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex flex-col items-center justify-center p-6 text-center"
          >
            <div className="w-16 h-16 rounded-full bg-blue-50 flex items-center justify-center mb-4">
              <PieChart className="h-8 w-8 text-blue-400" />
            </div>
            <h3 className="text-lg font-semibold text-slate-700 mb-2">No Sentiment Data</h3>
            <p className="text-sm text-slate-500 max-w-[250px] mb-4">
              Upload dataset files to analyze sentiment distribution across disaster events
            </p>
            <div className="grid grid-cols-5 gap-2 mt-2">
              {['Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral'].map((sentiment) => (
                <div 
                  key={sentiment}
                  className="flex flex-col items-center"
                >
                  <div 
                    className="w-5 h-5 rounded-full mb-1"
                    style={{ 
                      backgroundColor: sentiment === 'Panic' ? '#ef4444' :
                                    sentiment === 'Fear/Anxiety' ? '#f97316' :
                                    sentiment === 'Disbelief' ? '#8b5cf6' :
                                    sentiment === 'Resilience' ? '#10b981' : '#6b7280'
                    }}
                  ></div>
                  <span className="text-xs text-slate-500 whitespace-nowrap">
                    {sentiment.replace('/Anxiety', '')}
                  </span>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      )}

      {/* Chart Container */}
      <AnimatePresence mode="wait">
        {!isEmpty && !isLoading && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            className="h-full relative"
          >
            <canvas ref={chartRef}></canvas>
            
            {/* Center content for doughnut chart */}
            {type === 'doughnut' && processedData.values.length > 0 && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="text-center">
                  <div className="text-3xl font-bold text-slate-800">
                    {processedData.values.reduce((sum, val) => sum + val, 0)}
                  </div>
                  <div className="text-xs text-slate-500">Total Posts</div>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}