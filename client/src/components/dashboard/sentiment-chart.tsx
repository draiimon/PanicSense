import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { chartColors } from "@/lib/colors";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Pie } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip, Legend);

interface SentimentChartProps {
  data?: {
    labels: string[];
    values: number[];
  };
  title?: string;
}

export function SentimentChart({ 
  data = { labels: [], values: [] },
  title = "Sentiment Distribution" 
}: SentimentChartProps) {

  if (!data || !data.labels || !data.values) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-[300px]">
          <p className="text-gray-500">No sentiment data available</p>
        </CardContent>
      </Card>
    );
  }

  const chartData = {
    labels: data.labels,
    datasets: [{
      data: data.values,
      backgroundColor: chartColors.slice(0, data.labels.length),
      borderWidth: 0
    }]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'right' as const
      }
    }
  };

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[300px] flex items-center justify-center">
          <Pie data={chartData} options={options} />
        </div>
      </CardContent>
    </Card>
  );
}