import React, { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer,
  Cell,
  LabelList,
  CartesianGrid
} from "recharts";
import { ChartConfig } from "@/lib/chart-config";
import { 
  BarChart4, 
  Sparkles
} from "lucide-react";
import { Button } from "../ui/button";
import { motion } from "framer-motion";

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
  const [isAnimating, setIsAnimating] = useState(false);

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
    
    return Object.entries(counts)
      .map(([name, value]) => ({ 
        name, 
        value,
        // Calculate percentage for visual display
        percentage: data.length ? Math.round((value / data.length) * 100) : 0
      }));
  }, [data]);

  // Enhanced colors for each sentiment category with a gradient effect
  const sentimentColors = {
    "Panic": 'url(#panicGradient)',
    "Fear/Anxiety": 'url(#anxietyGradient)',
    "Disbelief": 'url(#disbeliefGradient)',
    "Neutral": 'url(#neutralGradient)',
    "Resilience": 'url(#resilienceGradient)'
  };

  // Updated solid colors as requested
  const sentimentSolidColors = {
    "Panic": "#ef4444", // Red
    "Fear/Anxiety": "#f97316", // Orange
    "Disbelief": "#8b5cf6", // Purple
    "Neutral": "#6b7280", // Gray
    "Resilience": "#22c55e" // Green (unchanged)
  };

  // Emoji icons for each sentiment category
  const sentimentEmojis = {
    "Panic": "😱",      // Face screaming in fear
    "Fear/Anxiety": "😨", // Fearful face
    "Disbelief": "😲",    // Astonished face
    "Neutral": "😐",      // Neutral face
    "Resilience": "💪"    // Flexed biceps (strength)
  };

  const totalRecords = sentimentCounts.reduce((sum, item) => sum + item.value, 0);

  // Enhanced tooltip with animation
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const percentage = ((data.value / totalRecords) * 100).toFixed(1);
      
      return (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="custom-tooltip bg-white p-4 border border-slate-200 rounded-md shadow-lg"
        >
          <div className="flex items-center gap-2 pb-2 border-b border-slate-100">
            <span className="text-xl">{sentimentEmojis[data.name as keyof typeof sentimentEmojis]}</span>
            <span className="font-medium text-base">{data.name}</span>
          </div>
          <div className="pt-2 space-y-1">
            <p className="text-sm flex justify-between">
              <span className="text-slate-500">Count:</span>
              <span className="font-semibold">{data.value}</span>
            </p>
            <p className="text-sm flex justify-between">
              <span className="text-slate-500">Percentage:</span>
              <span className="font-semibold">{percentage}%</span>
            </p>
          </div>
          <div 
            className="mt-2 h-1.5 w-full rounded-full bg-slate-100 overflow-hidden"
          >
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${percentage}%` }}
              transition={{ duration: 0.5 }}
              className="h-full rounded-full"
              style={{ backgroundColor: sentimentSolidColors[data.name as keyof typeof sentimentSolidColors] }}
            />
          </div>
        </motion.div>
      );
    }
    return null;
  };

  // Function to trigger animation effect
  const triggerAnimation = () => {
    setIsAnimating(true);
    setTimeout(() => setIsAnimating(false), 1500);
  };

  // Enhanced rendering of the bar chart with animation and custom styling
  const renderBarChart = () => (
    <div className="w-full h-[300px] sentiment-chart">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart 
          data={sentimentCounts} 
          margin={{ top: 20, right: 10, left: 0, bottom: 40 }}
        >
          <defs>
            {/* Define gradients for each sentiment category */}
            <linearGradient id="panicGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f87171" />
              <stop offset="100%" stopColor="#dc2626" />
            </linearGradient>
            <linearGradient id="anxietyGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#fb923c" />
              <stop offset="100%" stopColor="#ea580c" />
            </linearGradient>
            <linearGradient id="disbeliefGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#a78bfa" />
              <stop offset="100%" stopColor="#7c3aed" />
            </linearGradient>
            <linearGradient id="neutralGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#9ca3af" />
              <stop offset="100%" stopColor="#4b5563" />
            </linearGradient>
            <linearGradient id="resilienceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#4ade80" />
              <stop offset="100%" stopColor="#16a34a" />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis 
            dataKey="name" 
            angle={-45} 
            textAnchor="end" 
            tick={{ fontSize: 12, fill: '#64748b' }}
            tickMargin={10}
            axisLine={{ stroke: '#e2e8f0' }}
            tickLine={{ stroke: '#e2e8f0' }}
          />
          <YAxis 
            allowDecimals={false} 
            axisLine={{ stroke: '#e2e8f0' }}
            tickLine={{ stroke: '#e2e8f0' }}
            tick={{ fontSize: 12, fill: '#64748b' }}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(224, 231, 255, 0.2)' }} />
          <Bar 
            dataKey="value" 
            fill={ChartConfig.colors.primary} 
            radius={[6, 6, 0, 0]}
            animationDuration={isAnimating ? 1500 : 500} 
            animationBegin={0}
            animationEasing="ease-out"
          >
            <LabelList 
              dataKey="value" 
              position="top" 
              fill="#64748b" 
              fontSize={12} 
              fontWeight={600}
              formatter={(value: number) => (value > 0 ? value : '')}
            />
            {sentimentCounts.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={sentimentColors[entry.name as keyof typeof sentimentColors] || ChartConfig.colors.primary} 
                stroke={sentimentSolidColors[entry.name as keyof typeof sentimentSolidColors] || ChartConfig.colors.primary}
                strokeWidth={1}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );

  // Render visual indicator with enhanced styling for each sentiment category
  const renderSentimentLegend = () => (
    <div className="flex lg:flex-col gap-2 mt-4 lg:mt-0 lg:ml-6 flex-wrap justify-center">
      {sentimentCounts.map((item, index) => {
        const percentage = totalRecords > 0 ? Math.round((item.value / totalRecords) * 100) : 0;
        
        return (
          <motion.div 
            key={item.name} 
            className="flex items-center gap-2 px-4 py-3 rounded-lg bg-white border border-slate-200 shadow-sm hover:shadow-md transition-shadow"
            whileHover={{ scale: 1.02 }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div 
              className="h-4 w-4 rounded-full" 
              style={{ backgroundColor: sentimentSolidColors[item.name as keyof typeof sentimentSolidColors] || ChartConfig.colors.primary }}
            />
            <div className="flex flex-col">
              <div className="text-sm font-medium flex items-center gap-1.5">
                <span className="text-lg">{sentimentEmojis[item.name as keyof typeof sentimentEmojis]}</span>
                <span>{item.name}</span>
                <span className="font-bold">({item.value})</span>
              </div>
              <div className="mt-1 w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                <motion.div 
                  className="h-full" 
                  style={{ backgroundColor: sentimentSolidColors[item.name as keyof typeof sentimentSolidColors] || ChartConfig.colors.primary }}
                  initial={{ width: 0 }}
                  animate={{ width: isAnimating ? `${percentage}%` : `${percentage}%` }}
                  transition={{ duration: isAnimating ? 1 : 0.5, delay: isAnimating ? index * 0.1 : 0 }}
                />
              </div>
              <div className="text-xs text-slate-500 mt-1">{percentage}%</div>
            </div>
          </motion.div>
        );
      })}
    </div>
  );

  return (
    <Card className="h-full overflow-hidden border-slate-200 shadow-sm hover:shadow-md transition-shadow">
      <CardHeader className="pb-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart4 className="h-5 w-5 text-indigo-500" />
            <CardTitle className="text-base font-medium">{title}</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={triggerAnimation} 
              className="h-8 px-2"
              title="Animate Chart"
            >
              <Sparkles className="h-4 w-4 text-amber-500" />
            </Button>
          </div>
        </div>
        <p className="text-sm text-slate-500 mt-2">{description}</p>
      </CardHeader>
      <CardContent className="p-6">
        <div className="flex flex-col lg:flex-row justify-between">
          {renderBarChart()}
          {renderSentimentLegend()}
        </div>
      </CardContent>
    </Card>
  );
};