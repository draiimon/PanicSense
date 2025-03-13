import { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { analyzeText } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
import { getSentimentBadgeClasses } from '@/lib/colors';

interface AnalyzedText {
  text: string;
  sentiment: string;
  confidence: number;
  explanation: string;
  timestamp: Date;
}

export function RealtimeMonitor() {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzedTexts, setAnalyzedTexts] = useState<AnalyzedText[]>([]);
  const { toast } = useToast();

  // Auto-scroll to bottom of results
  const resultsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (resultsEndRef.current) {
      resultsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [analyzedTexts]);

  const handleAnalyze = async () => {
    if (!text.trim()) {
      toast({
        title: 'Empty text',
        description: 'Please enter some text to analyze',
        variant: 'destructive',
      });
      return;
    }

    setIsAnalyzing(true);
    try {
      const response = await fetch('/api/analyze-text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze text');
      }

      setAnalyzedTexts(prev => [
        ...prev,
        {
          text,
          sentiment: data.post.sentiment,
          confidence: data.post.confidence,
          explanation: data.post.explanation || "No explanation available",
          timestamp: new Date(),
        },
      ]);

      setText('');

      toast({
        title: 'Analysis complete',
        description: `Sentiment detected: ${data.post.sentiment}`,
      });
    } catch (error) {
      toast({
        title: 'Analysis failed',
        description: error instanceof Error ? error.message : 'An unexpected error occurred',
        variant: 'destructive',
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="grid gap-6 md:grid-cols-2">
      {/* Input Card */}
      <Card className="bg-white rounded-lg shadow">
        <CardHeader className="p-5 border-b border-gray-200">
          <CardTitle className="text-lg font-medium text-slate-800">Real-Time Sentiment Analysis</CardTitle>
          <CardDescription className="text-sm text-slate-500">
            Enter text related to disaster situations
          </CardDescription>
        </CardHeader>
        <CardContent className="p-5">
          <Textarea
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Enter disaster-related text to analyze sentiment..."
            className="min-h-[200px]"
          />
        </CardContent>
        <CardFooter className="p-5 pt-0 flex justify-end">
          <Button
            onClick={handleAnalyze}
            disabled={isAnalyzing || !text.trim()}
            className="bg-blue-600 hover:bg-blue-700"
          >
            {isAnalyzing ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Analyzing...
              </>
            ) : (
              'Analyze Sentiment'
            )}
          </Button>
        </CardFooter>
      </Card>

      {/* Results Card */}
      <Card className="bg-white rounded-lg shadow">
        <CardHeader className="p-5 border-b border-gray-200">
          <CardTitle className="text-lg font-medium text-slate-800">Analysis Results</CardTitle>
          <CardDescription className="text-sm text-slate-500">
            {analyzedTexts.length === 0
              ? 'No results yet - analyze some text to see results'
              : `Showing ${analyzedTexts.length} analyzed text${analyzedTexts.length !== 1 ? 's' : ''}`
            }
          </CardDescription>
        </CardHeader>
        <CardContent className="p-5 max-h-[500px] overflow-y-auto">
          {analyzedTexts.length === 0 ? (
            <div className="text-center py-10 text-slate-400">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-12 w-12 mx-auto mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <p className="font-medium">No analysis results yet</p>
              <p className="text-sm mt-1">Enter some text and click Analyze</p>
            </div>
          ) : (
            <div className="space-y-4">
              {analyzedTexts.map((item, index) => (
                <div key={index} className="p-4 bg-slate-50 rounded-lg">
                  <div className="flex justify-between items-start">
                    <p className="text-sm text-slate-900">{item.text}</p>
                    <Badge
                      className={getSentimentBadgeClasses(item.sentiment)}
                    >
                      {item.sentiment}
                    </Badge>
                  </div>
                  <div className="mt-2 flex justify-between text-xs text-slate-500">
                    <span>Confidence: {(item.confidence * 100).toFixed(1)}%</span>
                    <span>{item.timestamp.toLocaleTimeString()}</span>
                  </div>
                  <div className="mt-2 text-sm text-slate-600 italic">
                    <span className="font-medium">Analysis:</span> {item.explanation}
                  </div>
                </div>
              ))}
              <div ref={resultsEndRef} />
            </div>
          )}
        </CardContent>
        {analyzedTexts.length > 0 && (
          <CardFooter className="p-5 pt-0 flex justify-between">
            <Button
              variant="outline"
              onClick={() => setAnalyzedTexts([])}
              className="text-slate-600"
            >
              Clear Results
            </Button>
            <Button
              variant="ghost"
              className="text-blue-600"
              onClick={() => {
                const text = analyzedTexts.map(item =>
                  `"${item.text}" - ${item.sentiment} (${(item.confidence * 100).toFixed(1)}%) - ${item.explanation}`
                ).join('\n');
                navigator.clipboard.writeText(text);
                toast({
                  title: 'Copied to clipboard',
                  description: 'Analysis results have been copied to clipboard',
                });
              }}
            >
              Copy All
            </Button>
          </CardFooter>
        )}
      </Card>
    </div>
  );
}