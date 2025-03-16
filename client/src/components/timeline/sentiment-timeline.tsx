import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { chartColors, sentimentColors } from '@/lib/colors';
import Chart from 'chart.js/auto';
import 'chartjs-adapter-date-fns';
import { subDays, subWeeks, subMonths, parseISO, differenceInDays, format, isAfter, isEqual } from 'date-fns';

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
  rawDates?: string[]; // Original ISO date strings from posts
}

export function SentimentTimeline({ 
  data, 
  title = 'Sentiment Evolution',
  description = 'Last 7 days',
  rawDates = []
}: SentimentTimelineProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);
  const [timeRange, setTimeRange] = useState<'day' | 'week' | 'month'>('week');
  
  // Function to filter the data based on the selected time range
  const filterDataByTimeRange = () => {
    if (!rawDates || rawDates.length === 0) {
      return data; // Return original data if no raw dates
    }
    
    const currentDate = new Date();
    let cutoffDate: Date;
    
    // Determine cutoff date based on selected range
    switch (timeRange) {
      case 'day':
        cutoffDate = subDays(currentDate, 1);
        break;
      case 'week':
        cutoffDate = subWeeks(currentDate, 1);
        break;
      case 'month':
        cutoffDate = subMonths(currentDate, 1);
        break;
      default:
        cutoffDate = subDays(currentDate, 7);
    }
    
    // Convert all raw dates to Date objects for filtering
    const datePairs = rawDates.map(dateStr => {
      const date = parseISO(dateStr);
      const formattedDate = format(date, 'MMM dd, yyyy');
      return { original: dateStr, formatted: formattedDate, date };
    });
    
    // Filter for dates within the selected range
    const filteredDates = datePairs.filter(
      pair => isAfter(pair.date, cutoffDate) || isEqual(pair.date, cutoffDate)
    );
    
    // Get filtered formatted dates for labels
    const filteredLabels = [...new Set(filteredDates.map(pair => pair.formatted))];
    
    // Sort chronologically
    filteredLabels.sort((a, b) => {
      const dateA = parseISO(datePairs.find(pair => pair.formatted === a)?.original || '');
      const dateB = parseISO(datePairs.find(pair => pair.formatted === b)?.original || '');
      return dateA.getTime() - dateB.getTime();
    });
    
    // Create new datasets with filtered data
    const filteredDatasets = data.datasets.map(dataset => {
      const newData = filteredLabels.map(label => {
        const originalIndex = data.labels.indexOf(label);
        return originalIndex >= 0 ? dataset.data[originalIndex] : 0;
      });
      
      return {
        ...dataset,
        data: newData
      };
    });
    
    return {
      labels: filteredLabels,
      datasets: filteredDatasets
    };
  };
  
  const filteredData = filterDataByTimeRange();

  useEffect(() => {
    if (chartRef.current) {
      // Destroy previous chart if it exists
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }

      const ctx = chartRef.current.getContext('2d');
      if (!ctx) return;

      // Format datasets with sentiment-specific colors
      const formattedDatasets = filteredData.datasets.map((dataset, index) => {
        const sentimentColor = sentimentColors[dataset.label] || chartColors[index % chartColors.length];
        
        return {
          ...dataset,
          borderColor: sentimentColor,
          backgroundColor: `${sentimentColor}20`,
          fill: true,
          tension: 0.4
        };
      });

      // Create chart
      chartInstance.current = new Chart(ctx, {
        type: 'line',
        data: {
          labels: filteredData.labels,
          datasets: formattedDatasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              title: {
                display: true,
                text: 'Sentiment Percentage (%)'
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
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  let label = context.dataset.label || '';
                  if (label) {
                    label += ': ';
                  }
                  if (context.parsed.y !== null) {
                    label += Math.round(context.parsed.y * 10) / 10 + '%';
                  }
                  return label;
                }
              }
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
  }, [filteredData, timeRange]);

  // Get the correct description based on the time range
  const getRangeDescription = () => {
    if (filteredData.labels.length === 0) {
      return "No data available";
    }
    
    const count = filteredData.labels.length;
    switch(timeRange) {
      case 'day':
        return `Last 24 hours (${count} data point${count !== 1 ? 's' : ''})`;
      case 'week':
        return `Last 7 days (${count} data point${count !== 1 ? 's' : ''})`;
      case 'month':
        return `Last 30 days (${count} data point${count !== 1 ? 's' : ''})`;
      default:
        return `${count} data point${count !== 1 ? 's' : ''}`;
    }
  };

  return (
    <Card className="bg-white rounded-lg shadow">
      <CardHeader className="p-4 md:p-5 border-b border-gray-200 flex flex-col md:flex-row md:items-center md:justify-between space-y-3 md:space-y-0">
        <div>
          <CardTitle className="text-lg font-medium text-slate-800">{title}</CardTitle>
          <CardDescription className="text-sm text-slate-500">{getRangeDescription()}</CardDescription>
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
        {filteredData.labels.length === 0 ? (
          <div className="h-80 flex items-center justify-center text-slate-500">
            No data available for the selected time range
          </div>
        ) : (
          <div className="h-80">
            <canvas ref={chartRef} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
