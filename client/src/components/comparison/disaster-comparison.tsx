import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { chartColors, getDisasterTypeColor } from '@/lib/colors';
import Chart from 'chart.js/auto';
import { motion } from 'framer-motion';
import { Badge } from '@/components/ui/badge';


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

const containerVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.5 } }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } }
};

export function DisasterComparison({ 
  disasters, 
  title = 'Disaster Type Comparison',
  description = 'Sentiment distribution across different disasters'
}: DisasterComparisonProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);
  const [selectedDisasters, setSelectedDisasters] = useState<string[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);

  const sentimentLabels = ['Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral'];

  // Initialize with first two disasters if available
  useEffect(() => {
    if (disasters.length > 0 && selectedDisasters.length === 0) {
      setSelectedDisasters(disasters.slice(0, Math.min(2, disasters.length)).map(d => d.type));
    }
    setIsLoaded(true); // Set isLoaded to true after initial data load
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
          },
          animation: {
            duration: 1000,
            easing: 'easeInOutQuart'
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

  const toggleDisaster = (disasterType: string) => {
    setSelectedDisasters(prev => {
      if (prev.includes(disasterType)) {
        return prev.filter(d => d !== disasterType);
      } else {
        return [...prev, disasterType];
      }
    });
  };

  return (
    <motion.div
      initial="hidden"
      animate={isLoaded ? "visible" : "hidden"}
      variants={containerVariants}
    >
      <Card className="bg-white rounded-lg shadow overflow-hidden">
        <CardHeader className="p-5 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-slate-50">
          <motion.div variants={itemVariants}>
            <CardTitle className="text-lg font-medium text-slate-800">{title}</CardTitle>
            <CardDescription className="text-sm text-slate-500">{description}</CardDescription>
          </motion.div>
        </CardHeader>
        <CardContent className="p-5">
          {disasters.length === 0 ? (
            <motion.div 
              className="py-10 text-center text-slate-500"
              variants={itemVariants}
            >
              No disaster data available for comparison
            </motion.div>
          ) : (
            <div className="space-y-6">
              {/* Disaster Type Selector */}
              <motion.div variants={itemVariants}>
                <p className="mb-2 text-sm font-medium text-slate-700">Select disasters to compare:</p>
                <div className="flex flex-wrap gap-2">
                  {disasters.map((disaster, index) => (
                    <motion.div
                      key={disaster.type}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.1, duration: 0.3 }}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Badge 
                        variant={selectedDisasters.includes(disaster.type) ? "default" : "outline"}
                        className="cursor-pointer"
                        onClick={() => toggleDisaster(disaster.type)}
                      >
                        {disaster.type}
                      </Badge>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
              {/* Comparison Chart */}
              <motion.div 
                className="mt-4 p-2 rounded-lg bg-gradient-to-b from-white to-slate-50"
                variants={itemVariants}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4, duration: 0.5 }}
              >
                <canvas ref={chartRef} height={300}></canvas>
              </motion.div>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}