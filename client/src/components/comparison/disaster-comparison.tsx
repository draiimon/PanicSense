import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { chartColors } from '@/lib/colors';
import Chart from 'chart.js/auto';

interface DisasterData {
  type: string;
  sentiments: {
    label: string;
    percentage: number;
  }[];
}

interface DisasterComparisonProps {
  disasters: DisasterData[];
  title?: string;
  description?: string;
}

export function DisasterComparison({ 
  disasters, 
  title = 'Disaster Type Comparison',
  description = 'Sentiment distribution across different disasters'
}: DisasterComparisonProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);
  const [selectedDisasters, setSelectedDisasters] = useState<string[]>([]);

  const sentimentLabels = ['Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral'];

  // Initialize with first two disasters if available
  useEffect(() => {
    if (disasters.length > 0 && selectedDisasters.length === 0) {
      setSelectedDisasters(disasters.slice(0, Math.min(2, disasters.length)).map(d => d.type));
    }
  }, [disasters]);

  // Update chart when selection changes
  useEffect(() => {
    if (chartRef.current && selectedDisasters.length > 0) {
      // Destroy previous chart if it exists
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }

      const ctx = chartRef.current.getContext('2d');
      if (!ctx) return;

      // Filter selected disasters
      const selectedData = disasters.filter(d => selectedDisasters.includes(d.type));
      
      // Prepare data for chart
      const datasets = selectedData.map((disaster, index) => {
        // Create an array for each sentiment label
        const data = sentimentLabels.map(label => {
          const sentiment = disaster.sentiments.find(s => s.label === label);
          return sentiment ? sentiment.percentage : 0;
        });

        return {
          label: disaster.type,
          data,
          backgroundColor: chartColors[index % chartColors.length],
          borderColor: chartColors[index % chartColors.length],
          borderWidth: 1
        };
      });

      // Create chart
      chartInstance.current = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: sentimentLabels,
          datasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: 'Percentage'
              }
            },
            x: {
              title: {
                display: true,
                text: 'Sentiment'
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
  }, [disasters, selectedDisasters]);

  const handleDisasterToggle = (disasterType: string) => {
    setSelectedDisasters(prev => {
      if (prev.includes(disasterType)) {
        return prev.filter(d => d !== disasterType);
      } else {
        return [...prev, disasterType];
      }
    });
  };

  return (
    <Card className="bg-white rounded-lg shadow">
      <CardHeader className="p-5 border-b border-gray-200">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <div>
            <CardTitle className="text-lg font-medium text-slate-800">{title}</CardTitle>
            <CardDescription className="text-sm text-slate-500">{description}</CardDescription>
          </div>
          <div className="mt-4 md:mt-0">
            <Select
              onValueChange={(value) => setSelectedDisasters([value])}
              value={selectedDisasters[0]}
            >
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Select disaster" />
              </SelectTrigger>
              <SelectContent>
                {disasters.map((disaster) => (
                  <SelectItem key={disaster.type} value={disaster.type}>
                    {disaster.type}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-5">
        <div className="h-80">
          <canvas ref={chartRef} />
        </div>
        
        {/* Disaster Type Selection */}
        <div className="mt-6 flex flex-wrap gap-2">
          {disasters.map((disaster, index) => (
            <button
              key={disaster.type}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                selectedDisasters.includes(disaster.type) 
                  ? `bg-[${chartColors[index % chartColors.length]}] text-white` 
                  : 'bg-white border border-slate-300 text-slate-700'
              }`}
              style={{
                backgroundColor: selectedDisasters.includes(disaster.type) 
                  ? chartColors[index % chartColors.length] 
                  : undefined
              }}
              onClick={() => handleDisasterToggle(disaster.type)}
            >
              {disaster.type}
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
