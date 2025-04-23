import React, { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from "recharts";
import { ChartConfig } from "@/lib/chart-config";
import { BarChart4, HandHeart, Brain, Package, Shield } from "lucide-react";

interface SentimentLabelCountProps {
  data: any[];
  title?: string;
  description?: string;
}

export const SentimentLabelCount: React.FC<SentimentLabelCountProps> = ({
  data,
  title = "Sentiment Distribution",
  description = "Distribution of sentiment labels across the dataset"
}) => {
  const sentimentCounts = useMemo(() => {
    if (!Array.isArray(data)) return [];
    
    const counts: Record<string, number> = {
      "Panic": 0, 
      "Fear/Anxiety": 0, 
      "Disbelief": 0, 
      "Neutral": 0, 
      "Resilience": 0
    };
    
    data.forEach(item => {
      if (item.sentiment && counts[item.sentiment] !== undefined) {
        counts[item.sentiment] += 1;
      }
    });
    
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [data]);

  // Colors for each sentiment category
  const sentimentColors = {
    "Panic": ChartConfig.colors.red,
    "Fear/Anxiety": ChartConfig.colors.orange,
    "Disbelief": ChartConfig.colors.yellow,
    "Neutral": ChartConfig.colors.blue,
    "Resilience": ChartConfig.colors.green
  };

  // Icons for each sentiment category
  const sentimentIcons = {
    "Panic": <BarChart4 className="h-4 w-4" />,
    "Fear/Anxiety": <Brain className="h-4 w-4" />,
    "Disbelief": <Package className="h-4 w-4" />,
    "Neutral": <Shield className="h-4 w-4" />,
    "Resilience": <HandHeart className="h-4 w-4" />
  };

  const totalRecords = sentimentCounts.reduce((sum, item) => sum + item.value, 0);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const percentage = ((data.value / totalRecords) * 100).toFixed(1);
      
      return (
        <div className="custom-tooltip bg-white p-3 border border-slate-200 rounded-md shadow-md">
          <p className="flex items-center gap-2">
            {sentimentIcons[data.name]}
            <span className="font-medium">{data.name}</span>
          </p>
          <p className="text-sm mt-1">
            Count: <span className="font-medium">{data.value}</span>
          </p>
          <p className="text-sm">
            Percentage: <span className="font-medium">{percentage}%</span>
          </p>
        </div>
      );
    }
  
    return null;
  };

  return (
    <Card className="h-full overflow-hidden">
      <CardHeader className="pb-2">
        <CardTitle className="text-base font-medium">{title}</CardTitle>
        <p className="text-sm text-slate-500">{description}</p>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col lg:flex-row justify-between">
          <div className="w-full h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sentimentCounts} margin={{ top: 20, right: 10, left: 0, bottom: 40 }}>
                <XAxis 
                  dataKey="name" 
                  angle={-45} 
                  textAnchor="end" 
                  tick={{ fontSize: 12 }}
                  tickMargin={10}
                />
                <YAxis allowDecimals={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" fill={ChartConfig.colors.primary} radius={[4, 4, 0, 0]}>
                  {sentimentCounts.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={sentimentColors[entry.name] || ChartConfig.colors.primary} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex lg:flex-col gap-2 mt-4 lg:mt-0 lg:ml-6 flex-wrap justify-center">
            {sentimentCounts.map((item) => (
              <div 
                key={item.name} 
                className="flex items-center gap-2 px-3 py-2 rounded-md bg-slate-50 border border-slate-100"
              >
                <div 
                  className="h-3 w-3 rounded-full" 
                  style={{ backgroundColor: sentimentColors[item.name] || ChartConfig.colors.primary }}
                />
                <div className="text-xs font-medium flex items-center gap-1.5">
                  {sentimentIcons[item.name]}
                  <span>{item.name}</span>
                  <span className="font-bold">({item.value})</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};