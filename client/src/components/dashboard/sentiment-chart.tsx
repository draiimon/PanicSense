
import { motion } from "framer-motion";
import { useTheme } from "@/context/theme-context";
import { Bar, Line, Pie, Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// Register chart components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface SentimentChartProps {
  data: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      borderColor?: string;
      backgroundColor?: string | string[];
      fill?: boolean;
    }[];
  };
  title?: string;
  type: "line" | "bar" | "pie" | "doughnut";
  height?: number;
  description?: string;
}

export function SentimentChart({ 
  data, 
  title, 
  type, 
  height = 350,
  description
}: SentimentChartProps) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  // Define theme-dependent chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          boxWidth: 12,
          usePointStyle: true,
          pointStyle: 'circle',
          color: isDark ? 'rgba(255, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.7)',
          padding: 20,
          font: {
            family: 'Inter, sans-serif',
            size: 11,
          },
        },
      },
      tooltip: {
        backgroundColor: isDark ? 'rgba(15, 23, 42, 0.9)' : 'rgba(255, 255, 255, 0.95)',
        titleColor: isDark ? 'rgba(255, 255, 255, 0.9)' : 'rgba(0, 0, 0, 0.8)',
        bodyColor: isDark ? 'rgba(255, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.7)',
        padding: 12,
        boxPadding: 6,
        borderColor: isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        borderWidth: 1,
        boxWidth: 8,
        usePointStyle: true,
        callbacks: {
          // Format label based on chart type
          label: function(context: any) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== undefined) {
              label += context.parsed.y;
            } else if (context.parsed !== undefined) {
              label += context.parsed;
            }
            return label;
          }
        }
      },
    },
    scales: type === 'line' || type === 'bar' ? {
      x: {
        grid: {
          color: isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)',
        },
        ticks: {
          color: isDark ? 'rgba(255, 255, 255, 0.6)' : 'rgba(0, 0, 0, 0.6)',
          font: {
            size: 10,
          },
        },
      },
      y: {
        grid: {
          color: isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)',
        },
        ticks: {
          color: isDark ? 'rgba(255, 255, 255, 0.6)' : 'rgba(0, 0, 0, 0.6)',
          font: {
            size: 10,
          },
        },
        beginAtZero: true,
      },
    } : undefined,
  };

  // Animation variants for the chart container
  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.6,
        ease: "easeOut"
      }
    }
  };

  // Function to render the appropriate chart type
  const renderChart = () => {
    switch (type) {
      case "line":
        return <Line data={data} options={chartOptions} />;
      case "bar":
        return <Bar data={data} options={chartOptions} />;
      case "pie":
        return <Pie data={data} options={chartOptions} />;
      case "doughnut":
        return <Doughnut data={data} options={{...chartOptions, cutout: '70%'}} />;
      default:
        return <Bar data={data} options={chartOptions} />;
    }
  };

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={containerVariants}
      className="w-full"
    >
      <Card className={`overflow-hidden shadow-lg border ${
        isDark ? 'bg-slate-800 border-slate-700' : 'bg-white border-gray-100'
      }`}>
        {title && (
          <CardHeader className="pb-2">
            <CardTitle className={`text-lg font-semibold ${
              isDark ? 'text-white' : 'text-slate-800'
            }`}>
              {title}
            </CardTitle>
            {description && (
              <p className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-500'}`}>
                {description}
              </p>
            )}
          </CardHeader>
        )}
        <CardContent className={title ? 'pt-2' : 'pt-6'}>
          <div style={{ height: `${height}px` }} className="w-full relative">
            {/* Add subtle gradient overlay for aesthetics */}
            <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-transparent to-blue-500/5 pointer-events-none" />
            {renderChart()}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
