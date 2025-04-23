import React, { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { 
  PieChart, 
  Pie, 
  Cell, 
  Tooltip, 
  ResponsiveContainer,
  Legend
} from "recharts";
import { ChartConfig } from "@/lib/chart-config";
import { Globe } from "lucide-react";

interface LanguageLabelCountProps {
  data: any[];
  title?: string;
  description?: string;
}

export const LanguageLabelCount: React.FC<LanguageLabelCountProps> = ({
  data,
  title = "Language Distribution",
  description = "Distribution of language labels across the dataset"
}) => {
  const languageCounts = useMemo(() => {
    if (!Array.isArray(data)) return [];
    
    const counts: Record<string, number> = {};
    
    data.forEach(item => {
      if (item.language) {
        counts[item.language] = (counts[item.language] || 0) + 1;
      }
    });
    
    return Object.entries(counts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value); // Sort by count, descending
  }, [data]);

  const totalRecords = languageCounts.reduce((sum, item) => sum + item.value, 0);

  // Generate colors for chart segments
  const COLORS = [
    ChartConfig.colors.primary,
    ChartConfig.colors.blue,
    ChartConfig.colors.green,
    ChartConfig.colors.yellow,
    ChartConfig.colors.orange,
    ChartConfig.colors.red,
    ChartConfig.colors.violet,
    ChartConfig.colors.pink,
  ];

  const customTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const percentage = ((data.value / totalRecords) * 100).toFixed(1);
      
      return (
        <div className="bg-white p-3 border border-slate-200 rounded-md shadow-md">
          <p className="flex items-center gap-2 font-medium">{data.name}</p>
          <p className="text-sm">Count: <span className="font-medium">{data.value}</span></p>
          <p className="text-sm">Percentage: <span className="font-medium">{percentage}%</span></p>
        </div>
      );
    }
    return null;
  };

  const renderLabel = (entry: any) => {
    if ((entry.value / totalRecords) < 0.05) return null; // Skip small segments
    return entry.name;
  };

  return (
    <Card className="h-full overflow-hidden">
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <Globe className="h-5 w-5 text-blue-500" />
          <CardTitle className="text-base font-medium">{title}</CardTitle>
        </div>
        <p className="text-sm text-slate-500">{description}</p>
      </CardHeader>
      <CardContent>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={languageCounts}
                cx="50%"
                cy="50%"
                labelLine={true}
                label={renderLabel}
                outerRadius={100}
                fill={ChartConfig.colors.primary}
                dataKey="value"
              >
                {languageCounts.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={COLORS[index % COLORS.length]} 
                  />
                ))}
              </Pie>
              <Tooltip content={customTooltip} />
              <Legend
                layout="vertical"
                verticalAlign="middle"
                align="right"
                formatter={(value, entry, index) => (
                  <span className="text-sm">{value} ({languageCounts[index]?.value})</span>
                )}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};