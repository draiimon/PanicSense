
import { motion } from "framer-motion";
import { useTheme } from "@/context/theme-context";
import { Doughnut } from "react-chartjs-2";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface SentimentData {
  label: string;
  percentage: number;
}

interface DisasterData {
  type: string;
  sentiments: SentimentData[];
}

interface DisasterComparisonProps {
  disasters: DisasterData[];
  title: string;
  description: string;
}

export function DisasterComparison({ disasters, title, description }: DisasterComparisonProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  // Color palette for sentiments based on theme
  const sentimentColors = {
    "Panic": isDark ? "rgba(239, 68, 68, 0.85)" : "rgba(239, 68, 68, 0.75)",
    "Fear/Anxiety": isDark ? "rgba(245, 158, 11, 0.85)" : "rgba(245, 158, 11, 0.75)",
    "Disbelief": isDark ? "rgba(139, 92, 246, 0.85)" : "rgba(139, 92, 246, 0.75)",
    "Resilience": isDark ? "rgba(34, 197, 94, 0.85)" : "rgba(34, 197, 94, 0.75)",
    "Neutral": isDark ? "rgba(59, 130, 246, 0.85)" : "rgba(59, 130, 246, 0.75)",
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.3 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0, 
      transition: { 
        type: "spring", 
        stiffness: 100
      }
    }
  };

  const getChartData = (disaster: DisasterData) => {
    return {
      labels: disaster.sentiments.map(s => s.label),
      datasets: [
        {
          data: disaster.sentiments.map(s => s.percentage),
          backgroundColor: disaster.sentiments.map(s => sentimentColors[s.label as keyof typeof sentimentColors] || "rgba(156, 163, 175, 0.75)"),
          borderColor: isDark ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.1)",
          borderWidth: 1,
          hoverOffset: 15,
        },
      ],
    };
  };

  const chartOptions = {
    plugins: {
      legend: {
        display: true,
        position: 'bottom' as const,
        labels: {
          color: isDark ? "rgb(229, 231, 235)" : "rgb(75, 85, 99)",
          padding: 15,
          boxWidth: 12,
          font: {
            size: 11,
          }
        }
      },
      tooltip: {
        backgroundColor: isDark ? "rgba(30, 41, 59, 0.85)" : "rgba(255, 255, 255, 0.95)",
        titleColor: isDark ? "rgb(229, 231, 235)" : "rgb(17, 24, 39)",
        bodyColor: isDark ? "rgb(229, 231, 235)" : "rgb(75, 85, 99)",
        borderColor: isDark ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.1)",
        borderWidth: 1,
        padding: 10,
        boxWidth: 10,
        usePointStyle: true,
        callbacks: {
          label: function(context: any) {
            const label = context.label || '';
            const value = context.raw || 0;
            return `${label}: ${value.toFixed(1)}%`;
          }
        }
      }
    },
    cutout: '65%',
    responsive: true,
    maintainAspectRatio: false,
  };

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={containerVariants}
      className="mb-8"
    >
      <Card className={`shadow-xl ${isDark ? 'bg-slate-800 border-slate-700' : 'bg-white'}`}>
        <CardHeader className="pb-2">
          <motion.div variants={itemVariants}>
            <CardTitle className="text-2xl font-bold bg-gradient-to-r from-violet-600 to-indigo-600 bg-clip-text text-transparent">
              {title}
            </CardTitle>
            <CardDescription className={`${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
              {description}
            </CardDescription>
          </motion.div>
        </CardHeader>
        <CardContent className="pb-6">
          <motion.div 
            variants={itemVariants}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {disasters.map((disaster, index) => (
              <motion.div
                key={index}
                whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
                className={`p-4 rounded-xl relative overflow-hidden ${isDark ? 'bg-slate-700' : 'bg-gray-50'}`}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-transparent to-blue-500/5 pointer-events-none" />
                
                <h3 className="text-lg font-semibold mb-3 text-center">{disaster.type}</h3>
                
                <div className="h-48 mx-auto relative">
                  <Doughnut data={getChartData(disaster)} options={chartOptions} />
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <div className="text-center">
                      {(() => {
                        const dominantSentiment = disaster.sentiments.reduce((prev, current) => 
                          current.percentage > prev.percentage ? current : prev
                        );
                        return (
                          <>
                            <p className="text-xs uppercase tracking-wider opacity-75">Dominant</p>
                            <p className="font-bold">{dominantSentiment.label}</p>
                            <p className="text-sm">{dominantSentiment.percentage.toFixed(1)}%</p>
                          </>
                        );
                      })()}
                    </div>
                  </div>
                </div>

                <div className="mt-4 text-sm">
                  <div className="grid grid-cols-2 gap-1">
                    {disaster.sentiments.map((sentiment, i) => (
                      <div key={i} className="flex items-center justify-between">
                        <div className="flex items-center">
                          <div 
                            className="w-3 h-3 rounded-full mr-2" 
                            style={{ backgroundColor: sentimentColors[sentiment.label as keyof typeof sentimentColors] }}
                          />
                          <span className={`${isDark ? 'text-gray-300' : 'text-gray-700'} text-xs`}>
                            {sentiment.label}
                          </span>
                        </div>
                        <span className="font-medium text-xs">{sentiment.percentage.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </CardContent>
        <CardFooter className={`border-t ${isDark ? 'border-slate-700' : 'border-gray-100'} pt-4`}>
          <motion.p variants={itemVariants} className="text-sm text-gray-500 dark:text-gray-400">
            Data analyzed using ensemble neural networks with 95% classification accuracy
          </motion.p>
        </CardFooter>
      </Card>
    </motion.div>
  );
}
