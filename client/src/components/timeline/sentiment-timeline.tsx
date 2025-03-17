import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { chartColors, sentimentColors } from '@/lib/colors';
import Chart from 'chart.js/auto';
import 'chartjs-adapter-date-fns';
import { subDays, subWeeks, subMonths, parseISO, differenceInDays, format, isAfter, isEqual, getYear } from 'date-fns';
import { ChevronLeft, ChevronRight } from 'lucide-react';

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
  const [timeRange, setTimeRange] = useState<'day' | 'week' | 'month' | 'year'>('week');
  
  // Get all available years from rawDates
  const availableYears = rawDates.length > 0 
    ? [...new Set(rawDates.map(dateStr => getYear(parseISO(dateStr))))].sort((a, b) => a - b)
    : [new Date().getFullYear()];
  
  // State for selected year
  const [selectedYear, setSelectedYear] = useState<number>(
    availableYears.length > 0 ? availableYears[availableYears.length - 1] : new Date().getFullYear()
  );
  
  // Move to previous year if available
  const handlePreviousYear = () => {
    const currentIndex = availableYears.indexOf(selectedYear);
    if (currentIndex > 0) {
      setSelectedYear(availableYears[currentIndex - 1]);
    }
  };
  
  // Move to next year if available
  const handleNextYear = () => {
    const currentIndex = availableYears.indexOf(selectedYear);
    if (currentIndex < availableYears.length - 1) {
      setSelectedYear(availableYears[currentIndex + 1]);
    }
  };
  
  // Function to filter the data based on the selected time range and year
  const filterDataByTimeRangeAndYear = () => {
    if (!rawDates || rawDates.length === 0) {
      return data; // Return original data if no raw dates
    }
    
    const currentDate = new Date();
    let cutoffDate: Date;
    
    // Convert all raw dates to Date objects for filtering
    const datePairs = rawDates.map(dateStr => {
      const date = parseISO(dateStr);
      const formattedDate = format(date, 'MMM dd, yyyy');
      return { original: dateStr, formatted: formattedDate, date, year: getYear(date) };
    });
    
    // First filter by selected year
    let yearFilteredDates = datePairs.filter(pair => pair.year === selectedYear);
    
    // Then apply time range filter if needed
    let timeFilteredDates = yearFilteredDates;
    
    if (timeRange !== 'year') {
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
      
      // Only apply additional time filter if viewing current year
      if (selectedYear === new Date().getFullYear()) {
        timeFilteredDates = yearFilteredDates.filter(
          pair => isAfter(pair.date, cutoffDate) || isEqual(pair.date, cutoffDate)
        );
      }
    }
    
    // Get filtered formatted dates for labels
    const filteredLabels = [...new Set(timeFilteredDates.map(pair => pair.formatted))];
    
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
  
  const filteredData = filterDataByTimeRangeAndYear();

  useEffect(() => {
    if (chartRef.current) {
      // Destroy previous chart if it exists
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }

      const ctx = chartRef.current.getContext('2d');
      if (!ctx) return;

      // Format datasets with more vibrant sentiment-specific colors
      const formattedDatasets = filteredData.datasets.map((dataset, index) => {
        let color;
        
        // Get proper sentiment color
        switch(dataset.label) {
          case 'Panic':
            color = '#ef4444'; // Vibrant red
            break;
          case 'Fear/Anxiety':
            color = '#f97316'; // Vibrant orange
            break;
          case 'Disbelief':
            color = '#8b5cf6'; // Vibrant purple
            break;
          case 'Resilience':
            color = '#10b981'; // Vibrant green
            break;
          case 'Neutral':
            color = '#6b7280'; // Slate gray
            break;
          default:
            color = chartColors[index % chartColors.length];
        }
        
        return {
          ...dataset,
          borderColor: color,
          backgroundColor: `${color}20`, // 20% opacity
          borderWidth: 3,
          pointBackgroundColor: color,
          pointRadius: 4,
          pointHoverRadius: 6,
          fill: true,
          tension: 0.3 // Smoother lines
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
          animations: {
            tension: {
              duration: 1000,
              easing: 'linear',
              from: 0.8,
              to: 0.3,
              loop: false
            }
          },
          interaction: {
            mode: 'index',
            intersect: false
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              title: {
                display: true,
                text: 'Sentiment Percentage (%)',
                font: {
                  weight: 'bold'
                }
              },
              grid: {
                color: 'rgba(0, 0, 0, 0.05)',
                borderDash: [5, 5]
              },
              ticks: {
                callback: function(value) {
                  return value + '%';
                }
              }
            },
            x: {
              title: {
                display: true,
                text: 'Date',
                font: {
                  weight: 'bold'
                }
              },
              grid: {
                color: 'rgba(0, 0, 0, 0.05)',
                display: false
              }
            }
          },
          plugins: {
            legend: {
              position: 'bottom',
              labels: {
                usePointStyle: true,
                boxWidth: 8,
                boxHeight: 8,
                padding: 15,
                font: {
                  size: 12
                }
              }
            },
            tooltip: {
              backgroundColor: 'rgba(255, 255, 255, 0.9)',
              titleColor: '#1e293b',
              bodyColor: '#475569',
              borderColor: 'rgba(0, 0, 0, 0.1)',
              borderWidth: 1,
              padding: 10,
              boxPadding: 5,
              usePointStyle: true,
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
  }, [filteredData, timeRange, selectedYear]);

  // Get the correct description based on the time range and year
  const getRangeDescription = () => {
    if (filteredData.labels.length === 0) {
      return "No data available";
    }
    
    const count = filteredData.labels.length;
    const yearText = `${selectedYear}`;
    
    switch(timeRange) {
      case 'day':
        return `Last 24 hours, ${yearText} (${count} data point${count !== 1 ? 's' : ''})`;
      case 'week':
        return `Last 7 days, ${yearText} (${count} data point${count !== 1 ? 's' : ''})`;
      case 'month':
        return `Last 30 days, ${yearText} (${count} data point${count !== 1 ? 's' : ''})`;
      case 'year':
        return `Full year ${yearText} (${count} data point${count !== 1 ? 's' : ''})`;
      default:
        return `${count} data point${count !== 1 ? 's' : ''} for ${yearText}`;
    }
  };

  return (
    <Card className="bg-white rounded-lg shadow">
      <CardHeader className="p-4 md:p-5 border-b border-gray-200 flex flex-col space-y-3">
        <div className="flex flex-row justify-between items-center">
          <div>
            <CardTitle className="text-lg font-medium text-slate-800">{title}</CardTitle>
            <CardDescription className="text-sm text-slate-500">{getRangeDescription()}</CardDescription>
          </div>
          
          {/* Year selector */}
          <div className="flex items-center space-x-2">
            <Button 
              size="icon" 
              variant="outline" 
              onClick={handlePreviousYear}
              disabled={availableYears.indexOf(selectedYear) === 0}
              className="h-8 w-8"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            
            <div className="text-sm font-medium bg-slate-100 px-3 py-1 rounded min-w-[60px] text-center">
              {selectedYear}
            </div>
            
            <Button 
              size="icon" 
              variant="outline" 
              onClick={handleNextYear}
              disabled={availableYears.indexOf(selectedYear) === availableYears.length - 1}
              className="h-8 w-8"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        {/* Time range selector */}
        <div className="flex justify-end">
          <div className="flex space-x-2">
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
            <Button 
              size="sm" 
              variant={timeRange === 'year' ? 'default' : 'outline'}
              onClick={() => setTimeRange('year')}
              className="text-xs"
            >
              Year
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-5">
        {filteredData.labels.length === 0 ? (
          <div className="h-80 flex items-center justify-center text-slate-500">
            No data available for {selectedYear} in the selected time range
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
