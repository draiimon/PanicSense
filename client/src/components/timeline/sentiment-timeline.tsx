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
    ? Array.from(new Set(rawDates.map(dateStr => getYear(parseISO(dateStr))))).sort((a, b) => a - b)
    : [new Date().getFullYear()];

  // State for selected years
  const [selectedYears, setSelectedYears] = useState<number[]>(availableYears.length > 0 ? [availableYears[availableYears.length -1]] : [new Date().getFullYear()]);

  const selectAllYears = () => {
    setSelectedYears(availableYears);
  };

  const toggleYear = (year: number) => {
    setSelectedYears(prevYears => {
      if (prevYears.includes(year)) {
        return prevYears.filter(y => y !== year);
      } else {
        return [...prevYears, year];
      }
    });
  };

  const clearYears = () => {
    setSelectedYears([availableYears[availableYears.length-1]]);
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

    // First filter by selected years
    let yearFilteredDates = datePairs.filter(pair => selectedYears.includes(pair.year));

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
      timeFilteredDates = yearFilteredDates.filter(
        pair => isAfter(pair.date, cutoffDate) || isEqual(pair.date, cutoffDate)
      );
    }

    // Get filtered formatted dates for labels
    const filteredLabels = Array.from(new Set(timeFilteredDates.map(pair => pair.formatted)));

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
  }, [filteredData, timeRange, selectedYears]);

  // Get the correct description based on the time range and year
  const getRangeDescription = () => {
    if (filteredData.labels.length === 0) {
      return "No data available";
    }

    const count = filteredData.labels.length;
    const yearsText = selectedYears.length > 1 ? `Years: ${selectedYears.join(', ')}` : `Year: ${selectedYears[0]}`;

    switch(timeRange) {
      case 'day':
        return `Last 24 hours, ${yearsText} (${count} data point${count !== 1 ? 's' : ''})`;
      case 'week':
        return `Last 7 days, ${yearsText} (${count} data point${count !== 1 ? 's' : ''})`;
      case 'month':
        return `Last 30 days, ${yearsText} (${count} data point${count !== 1 ? 's' : ''})`;
      case 'year':
        return `Full year${selectedYears.length > 1 ? 's ' + selectedYears.join(', ') : ' ' + selectedYears[0]} (${count} data point${count !== 1 ? 's' : ''})`;
      default:
        return `${count} data point${count !== 1 ? 's' : ''} for ${yearsText}`;
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
          <div className="flex justify-between items-center">
            <div className="flex gap-2 items-center">
              <Button
                onClick={() => {
                  const currentIndex = availableYears.indexOf(selectedYears[0]);
                  if (currentIndex > 0) {
                    const prevYear = availableYears[currentIndex - 1];
                    setSelectedYears([prevYear]);
                  }
                }}
                variant="outline"
                size="sm"
                disabled={selectedYears.length !== 1 || selectedYears[0] === Math.min(...availableYears)}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="font-medium">
                {selectedYears.length === availableYears.length 
                  ? "All Years" 
                  : selectedYears.join(', ')}
              </span>
              <Button
                onClick={() => {
                  const currentIndex = availableYears.indexOf(selectedYears[0]);
                  if (currentIndex < availableYears.length - 1) {
                    const nextYear = availableYears[currentIndex + 1];
                    setSelectedYears([nextYear]);
                  }
                }}
                variant="outline"
                size="sm"
                disabled={selectedYears.length !== 1 || selectedYears[0] === Math.max(...availableYears)}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={selectAllYears}
                variant={selectedYears.length === availableYears.length ? "default" : "outline"}
                size="sm"
              >
                All Years
              </Button>
            </div>
          </div>
        </div>

        {/* Time range selector dropdown */}
        <div className="flex justify-end">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as 'day' | 'week' | 'month' | 'year')}
            className="bg-white border border-gray-200 text-slate-700 text-sm rounded-md p-2 pr-8 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          >
            <option value="day">Day</option>
            <option value="week">Week</option>
            <option value="month">Month</option>
            <option value="year">Year</option>
          </select>
        </div>
      </CardHeader>
      <CardContent className="p-5">
        {filteredData.labels.length === 0 ? (
          <div className="h-80 flex items-center justify-center text-slate-500">
            No data available for {selectedYears.join(', ')} in the selected time range
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