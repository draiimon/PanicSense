import React, { useMemo, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Cloud, Download } from "lucide-react";
import { Button } from "../ui/button";

interface WordCloudProps {
  data: any[];
  title?: string;
  description?: string;
}

interface WordFrequency {
  text: string;
  value: number;
}

export const WordCloud: React.FC<WordCloudProps> = ({
  data,
  title = "Word Cloud",
  description = "Visual representation of the most frequent words in the dataset"
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const wordFrequencies = useMemo(() => {
    if (!Array.isArray(data)) return [];
    
    // Common words to exclude (stopwords)
    const stopwords = new Set([
      'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 
      'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'could', 'did', 'do', 'does', 
      'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 
      'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 
      'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 
      'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should', 'so', 'some', 'such', 
      't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 
      'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 
      'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
    ]);
    
    // Tokenize and count word frequencies
    const wordCounts: Record<string, number> = {};
    
    data.forEach(item => {
      if (item.text) {
        const words = item.text
          .toLowerCase()
          .replace(/[^\w\s]/g, '') // Remove punctuation
          .split(/\s+/) // Split by whitespace
          .filter(word => word.length > 2 && !stopwords.has(word)); // Filter out stopwords and short words
        
        words.forEach(word => {
          wordCounts[word] = (wordCounts[word] || 0) + 1;
        });
      }
    });
    
    // Convert to array and sort by frequency
    return Object.entries(wordCounts)
      .map(([text, value]) => ({ text, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 100); // Limit to top 100 words
  }, [data]);

  useEffect(() => {
    if (!canvasRef.current || !containerRef.current || wordFrequencies.length === 0) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const container = containerRef.current;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Find max frequency to normalize
    const maxFrequency = Math.max(...wordFrequencies.map(w => w.value));
    
    // Draw words
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    const COLORS = [
      '#4f46e5', // primary indigo
      '#2563eb', // blue
      '#059669', // green
      '#ca8a04', // yellow
      '#ea580c', // orange
      '#dc2626', // red
      '#8b5cf6', // violet
      '#ec4899', // pink
    ];
    
    let placedWords: Array<{
      text: string;
      x: number;
      y: number;
      width: number;
      height: number;
    }> = [];
    
    // Helper to check if a word overlaps with already placed words
    const overlaps = (x: number, y: number, width: number, height: number) => {
      // Check canvas boundaries
      if (x - width/2 < 0 || x + width/2 > canvas.width || 
          y - height/2 < 0 || y + height/2 > canvas.height) {
        return true;
      }
      
      // Check other words
      for (const word of placedWords) {
        if (!(x + width/2 < word.x - word.width/2 || 
              x - width/2 > word.x + word.width/2 || 
              y + height/2 < word.y - word.height/2 || 
              y - height/2 > word.y + word.height/2)) {
          return true;
        }
      }
      
      return false;
    };
    
    // Place words using spiral placement algorithm
    wordFrequencies.forEach((word, idx) => {
      const scaleFactor = 0.5 + (word.value / maxFrequency) * 1.5;
      const fontSize = Math.max(12, Math.min(60, Math.floor(15 * scaleFactor)));
      
      ctx.font = `bold ${fontSize}px Inter, system-ui, sans-serif`;
      ctx.fillStyle = COLORS[idx % COLORS.length];
      
      const metrics = ctx.measureText(word.text);
      const width = metrics.width;
      const height = fontSize;
      
      // Spiral placement
      const angle = idx * 0.5;
      const radius = 5 + 10 * Math.sqrt(idx);
      let x = centerX + Math.cos(angle) * radius;
      let y = centerY + Math.sin(angle) * radius;
      
      // If overlapping, try different positions
      let attempts = 0;
      const maxAttempts = 200;
      
      while (overlaps(x, y, width, height) && attempts < maxAttempts) {
        attempts++;
        const newAngle = angle + attempts * 0.1;
        const newRadius = radius + attempts * 2;
        x = centerX + Math.cos(newAngle) * newRadius;
        y = centerY + Math.sin(newAngle) * newRadius;
      }
      
      if (attempts < maxAttempts) {
        ctx.fillText(word.text, x, y);
        placedWords.push({ text: word.text, x, y, width, height });
      }
    });
    
  }, [wordFrequencies]);

  // Function to handle downloading the word cloud as an image
  const handleDownload = () => {
    if (!canvasRef.current) return;
    
    // Create a temporary link element
    const link = document.createElement('a');
    link.download = 'word-cloud.png';
    
    // Convert canvas to data URL
    link.href = canvasRef.current.toDataURL('image/png');
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Card className="h-full overflow-hidden">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Cloud className="h-5 w-5 text-indigo-500" />
            <CardTitle className="text-base font-medium">{title}</CardTitle>
          </div>
          <Button variant="outline" size="sm" onClick={handleDownload} className="h-8">
            <Download className="h-4 w-4 mr-1" />
            Save Image
          </Button>
        </div>
        <p className="text-sm text-slate-500">{description}</p>
      </CardHeader>
      <CardContent>
        <div 
          ref={containerRef} 
          className="h-[350px] w-full relative bg-slate-50 rounded-lg overflow-hidden"
        >
          <canvas ref={canvasRef} className="absolute inset-0" />
          {wordFrequencies.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center">
              <p className="text-slate-400">No data available for word cloud</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};