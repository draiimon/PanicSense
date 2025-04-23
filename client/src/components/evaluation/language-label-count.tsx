import React, { useMemo, useState } from "react";
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
import { Globe, Download, Sparkles } from "lucide-react";
import { Button } from "../ui/button";
import { motion } from "framer-motion";

interface LanguageLabelCountProps {
  data: any[];
  title?: string;
  description?: string;
}

export const LanguageLabelCount: React.FC<LanguageLabelCountProps> = ({
  data,
  title = "Language Distribution",
  description = "Distribution of languages across the dataset"
}) => {
  const [isAnimating, setIsAnimating] = useState(false);
  const [exploded, setExploded] = useState<number | null>(null);

  const languageCounts = useMemo(() => {
    if (!Array.isArray(data)) return [];
    
    const counts: Record<string, number> = {};
    
    data.forEach(item => {
      if (item.language) {
        const lang = item.language;
        counts[lang] = (counts[lang] || 0) + 1;
      }
    });
    
    // Sort languages by count
    return Object.entries(counts)
      .map(([name, value]) => ({ 
        name, 
        value,
        // Calculate percentage for visual display
        percentage: data.length ? Math.round((value / data.length) * 100) : 0
      }))
      .sort((a, b) => b.value - a.value); // Sort by highest count first
  }, [data]);

  // Language emoji mapping
  const languageEmojis: Record<string, string> = {
    "English": "🇬🇧",
    "Filipino": "🇵🇭",
    "Taglish": "🇵🇭🇬🇧",
    "Bisaya": "🏝️",
    "Cebuano": "🌊",
    "Waray": "🌴",
    "Hiligaynon": "🌺",
    "Ilocano": "🌄",
    "Spanish": "🇪🇸",
    "Unknown": "❓"
  };

  // Dynamic colors based on language
  const languageColors: Record<string, string> = {
    "English": ChartConfig.colors.blue,
    "Filipino": ChartConfig.colors.indigo as string,
    "Taglish": ChartConfig.colors.violet,
    "Bisaya": ChartConfig.colors.yellow,
    "Cebuano": ChartConfig.colors.green,
    "Waray": ChartConfig.colors.red,
    "Hiligaynon": ChartConfig.colors.orange,
    "Ilocano": ChartConfig.colors.cyan as string,
    "Spanish": ChartConfig.colors.pink,
    "Unknown": ChartConfig.colors.gray as string
  };

  // Total records count
  const totalRecords = languageCounts.reduce((sum, item) => sum + item.value, 0);

  // Custom tooltip component with animation
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
            <span className="text-xl">{languageEmojis[data.name] || "🌐"}</span>
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
              style={{ backgroundColor: languageColors[data.name] || ChartConfig.colors.blue }}
            />
          </div>
        </motion.div>
      );
    }
    return null;
  };

  // Custom legend renderer with emoji icons
  const renderCustomLegend = () => (
    <div className="flex flex-col gap-2 mt-4">
      {languageCounts.map((entry, index) => {
        const percentage = totalRecords > 0 ? Math.round((entry.value / totalRecords) * 100) : 0;
        
        return (
          <motion.div 
            key={entry.name} 
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white border border-slate-200 shadow-sm hover:shadow-md transition-shadow cursor-pointer"
            whileHover={{ scale: 1.02 }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.05 }}
            onClick={() => handleSectionClick(index)}
          >
            <div 
              className="h-3 w-3 rounded-full" 
              style={{ backgroundColor: languageColors[entry.name] || ChartConfig.colors.blue }}
            />
            <div className="flex flex-col">
              <div className="text-sm font-medium flex items-center gap-1.5">
                <span className="text-lg">{languageEmojis[entry.name] || "🌐"}</span>
                <span>{entry.name}</span>
                <span className="font-bold">({entry.value})</span>
              </div>
              <div className="mt-1 w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                <motion.div 
                  className="h-full" 
                  style={{ backgroundColor: languageColors[entry.name] || ChartConfig.colors.blue }}
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

  // Function to handle clicking on a pie section
  const handleSectionClick = (index: number) => {
    setExploded(exploded === index ? null : index);
  };

  // Function to calculate the offset for "exploded" pie sections
  const getExplodedOffset = (index: number) => {
    return exploded === index ? 20 : 0;
  };

  // Function to trigger animation effect
  const triggerAnimation = () => {
    setIsAnimating(true);
    setTimeout(() => setIsAnimating(false), 1500);
  };

  // Function to handle downloading the chart as an image
  const handleDownload = () => {
    const svgElement = document.querySelector('.language-chart svg');
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
      link.download = 'language-distribution.png';
      link.href = canvas.toDataURL('image/png');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };
    
    img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
  };

  return (
    <Card className="h-full overflow-hidden border-slate-200 shadow-sm hover:shadow-md transition-shadow">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Globe className="h-5 w-5 text-blue-500" />
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
        <p className="text-sm text-slate-500">{description}</p>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col lg:flex-row justify-between">
          <div className="w-full lg:w-1/2 h-[300px] language-chart">
            {languageCounts.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <defs>
                    {/* Create gradient definitions for each language */}
                    {languageCounts.map((entry, index) => (
                      <linearGradient 
                        key={`gradient-${index}`} 
                        id={`gradient-${index}`} 
                        x1="0" 
                        y1="0" 
                        x2="0" 
                        y2="1"
                      >
                        <stop 
                          offset="0%" 
                          stopColor={languageColors[entry.name] || ChartConfig.colors.blue} 
                          stopOpacity={0.9} 
                        />
                        <stop 
                          offset="100%" 
                          stopColor={languageColors[entry.name] || ChartConfig.colors.blue} 
                          stopOpacity={0.6} 
                        />
                      </linearGradient>
                    ))}
                  </defs>
                  <Pie
                    data={languageCounts}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    innerRadius={isAnimating ? 60 : 50}
                    outerRadius={90}
                    paddingAngle={3}
                    dataKey="value"
                    animationDuration={isAnimating ? 1500 : 500}
                    animationBegin={0}
                    animationEasing="ease-out"
                  >
                    {languageCounts.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={`url(#gradient-${index})`}
                        stroke={languageColors[entry.name] || ChartConfig.colors.blue}
                        strokeWidth={1}
                      />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <p className="text-slate-400">No language data available</p>
              </div>
            )}
          </div>
          
          <div className="mt-4 lg:mt-0 lg:w-1/2 ml-0 lg:ml-6">
            {renderCustomLegend()}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};