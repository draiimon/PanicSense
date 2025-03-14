import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface SentimentChartProps {
  data: {
    labels: string[];
    values: number[];
  };
}

const chartColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD'];

export function SentimentChart({ data }: SentimentChartProps) {
  const labels = data?.labels || [];
  const values = data?.values || [];

  const chartData = {
    labels,
    datasets: [{
      data: values,
      backgroundColor: chartColors.slice(0, labels.length),
      borderWidth: 0
    }]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        display: false
      }
    },
    scales: {
      y: {
        beginAtZero: true
      }
    }
  };

  return (
    <Card className="bg-white rounded-lg shadow">
      <CardHeader className="border-b border-gray-200">
        <CardTitle className="text-lg font-medium text-slate-800">Sentiment Distribution</CardTitle>
        <CardDescription className="text-sm text-slate-500">Distribution of sentiments across posts</CardDescription>
      </CardHeader>
      <CardContent className="p-6">
        <Bar data={chartData} options={options} height={300} />
      </CardContent>
    </Card>
  );
}