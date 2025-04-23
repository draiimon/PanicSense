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
  CartesianGrid,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis
} from "recharts";
import { ChartConfig } from "@/lib/chart-config";
import { 
  BarChart4, 
  Download,
  Sparkles, 
  PieChart as PieChartIcon
} from "lucide-react";
import { Button } from "../ui/button";
import { motion } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";

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
  const [viewMode, setViewMode] = useState<'bar' | 'radial'>('bar');

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

  // Solid colors for labels and legend
  const sentimentSolidColors = {
    "Panic": ChartConfig.colors.red,
    "Fear/Anxiety": ChartConfig.colors.orange,
    "Disbelief": ChartConfig.colors.yellow,
    "Neutral": ChartConfig.colors.blue,
    "Resilience": ChartConfig.colors.green
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

  // Function to handle download as image
  const handleDownload = () => {
    const svgElement = document.querySelector('.sentiment-chart svg');
    if (!svgElement) return;
    
    // Create a canvas element
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions to match the SVG
    canvas.width = svgElement.clientWidth * 2; // Scale up for better quality
    canvas.height = svgElement.clientHeight * 2;
    ctx.scale(2, 2);
    
    // Create an image from the SVG
    const svgData = new XMLSerializer().serializeToString(svgElement);
    const img = new Image();
    
    img.onload = () => {
      // Draw a white background
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw the image
      ctx.drawImage(img, 0, 0);
      
      // Add title and timestamp
      ctx.font = 'bold 16px Inter, sans-serif';
      ctx.fillStyle = '#333';
      ctx.textAlign = 'center';
      ctx.fillText(title, canvas.width / 4, 20);
      
      ctx.font = '12px Inter, sans-serif';
      ctx.fillStyle = '#666';
      ctx.fillText(`Generated on ${new Date().toLocaleDateString()}`, canvas.width / 4, 40);
      
      // Create download link
      const link = document.createElement('a');
      link.download = 'sentiment-distribution.png';
      link.href = canvas.toDataURL('image/png');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };
    
    img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
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
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="100%" stopColor="#b91c1c" />
            </linearGradient>
            <linearGradient id="anxietyGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f97316" />
              <stop offset="100%" stopColor="#c2410c" />
            </linearGradient>
            <linearGradient id="disbeliefGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#eab308" />
              <stop offset="100%" stopColor="#a16207" />
            </linearGradient>
            <linearGradient id="neutralGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" />
              <stop offset="100%" stopColor="#1e40af" />
            </linearGradient>
            <linearGradient id="resilienceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#22c55e" />
              <stop offset="100%" stopColor="#15803d" />
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

  // Render the radial chart
  const renderRadialChart = () => (
    <div className="w-full h-[350px] sentiment-chart">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart 
          data={sentimentCounts} 
          margin={{ top: 30, right: 30, bottom: 30, left: 30 }}
          layout="radial"
          barCategoryGap={15}
        >
          <defs>
            {/* Define gradients for each sentiment category */}
            <linearGradient id="panicRadial" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="100%" stopColor="#b91c1c" />
            </linearGradient>
            <linearGradient id="anxietyRadial" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#f97316" />
              <stop offset="100%" stopColor="#c2410c" />
            </linearGradient>
            <linearGradient id="disbeliefRadial" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#eab308" />
              <stop offset="100%" stopColor="#a16207" />
            </linearGradient>
            <linearGradient id="neutralRadial" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#3b82f6" />
              <stop offset="100%" stopColor="#1e40af" />
            </linearGradient>
            <linearGradient id="resilienceRadial" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#22c55e" />
              <stop offset="100%" stopColor="#15803d" />
            </linearGradient>
          </defs>
          <PolarGrid stroke="#e2e8f0" />
          <PolarAngleAxis 
            dataKey="name" 
            tick={{ fontSize: 12, fill: '#64748b' }}
          />
          <PolarRadiusAxis angle={90} domain={[0, 'auto']} />
          <Tooltip content={<CustomTooltip />} />
          <Bar 
            dataKey="value" 
            barSize={20}
            animationDuration={isAnimating ? 1500 : 500} 
            animationBegin={0}
            animationEasing="ease-out"
          >
            {sentimentCounts.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={`url(#${entry.name.toLowerCase().replace(/[^a-z]/g, '')}Radial)`} 
                stroke={sentimentSolidColors[entry.name as keyof typeof sentimentSolidColors] || ChartConfig.colors.primary}
                strokeWidth={1}
              />
            ))}
            <LabelList 
              dataKey="value" 
              position="outside" 
              fill="#64748b" 
              fontSize={12} 
              fontWeight={600}
              formatter={(value: number) => (value > 0 ? value : '')}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
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
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleDownload} 
              className="h-8"
              title="Download Chart"
            >
              <Download className="h-4 w-4 mr-1" />
              Save
            </Button>
          </div>
        </div>
        <p className="text-sm text-slate-500 mt-2">{description}</p>
      </CardHeader>
      <CardContent className="p-6">
        <Tabs 
          defaultValue="bar" 
          value={viewMode}
          onValueChange={(val) => setViewMode(val as 'bar' | 'radial')}
          className="mt-2"
        >
          <TabsList className="grid w-[180px] grid-cols-2 mb-6">
            <TabsTrigger value="bar" className="flex items-center gap-1">
              <BarChart4 className="h-3.5 w-3.5" />
              <span>Bar</span>
            </TabsTrigger>
            <TabsTrigger value="radial" className="flex items-center gap-1">
              <PieChartIcon className="h-3.5 w-3.5" />
              <span>Radial</span>
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="bar" className="flex flex-col lg:flex-row justify-between">
            {renderBarChart()}
            {renderSentimentLegend()}
          </TabsContent>
          
          <TabsContent value="radial">
            {renderRadialChart()}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};