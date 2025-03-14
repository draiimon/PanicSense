import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

const chartColors = [
  '#FF6384',
  '#36A2EB',
  '#FFCE56',
  '#4BC0C0',
  '#9966FF'
];

interface SentimentChartProps {
  data: {
    labels: string[];
    values: number[];
  };
}

export function SentimentChart({ data }: SentimentChartProps) {
  if (!data?.labels?.length || !data?.values?.length) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Sentiment Distribution</CardTitle>
        </CardHeader>
        <CardContent className="h-[300px] flex items-center justify-center">
          <p className="text-slate-500">No sentiment data available</p>
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

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Sentiment Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[300px] flex items-center justify-center">
          <Pie data={chartData} options={{ maintainAspectRatio: false }} />
        </div>
      </CardContent>
    </Card>
  );
}