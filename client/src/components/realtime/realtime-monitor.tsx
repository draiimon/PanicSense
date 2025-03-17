import { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { analyzeText } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
import { getSentimentBadgeClasses } from '@/lib/colors';
import { AlertCircle } from 'lucide-react';
import { useDisasterContext } from '@/context/disaster-context';

interface AnalyzedText {
  text: string;
  sentiment: string;
  confidence: number;
  timestamp: Date;
  language: string;
  explanation?: string | null;
  disasterType?: string | null;
  location?: string | null;
}

export function RealtimeMonitor() {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzedTexts, setAnalyzedTexts] = useState<AnalyzedText[]>([]);
  const [autoAnalyze, setAutoAnalyze] = useState(false);
  const [typingTimeout, setTypingTimeout] = useState<NodeJS.Timeout | null>(null);
  const { toast } = useToast();
  const { refreshData } = useDisasterContext();

  // Auto-scroll to bottom of results
  const resultsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (resultsEndRef.current) {
      resultsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [analyzedTexts]);

  // Effect for auto-analyze
  useEffect(() => {
    if (autoAnalyze && text.trim() && !isAnalyzing) {
      if (typingTimeout) {
        clearTimeout(typingTimeout);
      }

      const timeout = setTimeout(() => {
        handleAnalyze();
      }, 1000);

      setTypingTimeout(timeout);
    }

    return () => {
      if (typingTimeout) {
        clearTimeout(typingTimeout);
      }
    };
  }, [text, autoAnalyze]);

  const handleAnalyze = async () => {
    if (!text.trim()) {
      if (!autoAnalyze) {
        toast({
          title: 'Empty text',
          description: 'Please enter some text to analyze',
          variant: 'destructive',
        });
      }
      return;
    }

    setIsAnalyzing(true);
    try {
      // Clean the text by removing extra spaces but preserve emojis and punctuation
      const cleanedText = text.trim().replace(/\s+/g, ' ');

      const result = await analyzeText(cleanedText);

      // Detect Tagalog by checking if the text contains common Filipino words or if model detected it
      const hasTagalogWords = /\b(ang|ng|mga|sa|ko|mo|nang|para|nung|yung|at|pag|ni|si|kay|na|po|opo|din|rin|nga|ba|eh|ay|ito|iyan|iyon|dito|diyan|doon)\b/i.test(cleanedText.toLowerCase());
      const detectedLanguage = hasTagalogWords || result.post.language === 'tl' ? 'tl' : 'en';

      // Create the analyzed text entry
      const analyzedText: AnalyzedText = {
        text: cleanedText,
        sentiment: result.post.sentiment,
        confidence: result.post.confidence,
        timestamp: new Date(),
        language: detectedLanguage,
        explanation: result.post.explanation,
        disasterType: result.post.disasterType,
        location: result.post.location
      };

      setAnalyzedTexts(prev => [
        ...prev,
        analyzedText
      ]);

      setText('');

      // Check if this is a disaster-related input
      const isNonDisasterInput = cleanedText.length < 9 || 
                              !result.post.explanation || 
                              result.post.disasterType === "Not Specified" ||
                              !result.post.disasterType;

      if (isNonDisasterInput) {
        toast({
          title: 'Non-Disaster Input',
          description: 'This appears to be a short non-disaster related input. For best results, please enter more detailed text about disaster situations.',
          variant: 'destructive',
          duration: 5000,
        });
      } else if (!autoAnalyze) {
        toast({
          title: 'Analysis complete',
          description: `Language: ${detectedLanguage === 'tl' ? 'Tagalog' : 'English'}, Sentiment: ${result.post.sentiment}`,
        });
      }

      // Refresh the data in the disaster context
      refreshData();
    } catch (error) {
      console.error('Analysis error:', error);
      toast({
        title: 'Analysis failed',
        description: 'Error processing text. Please try again.',
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
          <div className="flex justify-between items-center">
            <div>
              <CardTitle className="text-lg font-medium text-slate-800">Real-Time Sentiment Analysis</CardTitle>
              <CardDescription className="text-sm text-slate-500">
                Enter text related to disaster situations
              </CardDescription>
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="auto-analyze"
                checked={autoAnalyze}
                onCheckedChange={setAutoAnalyze}
              />
              <Label htmlFor="auto-analyze">Auto Analyze</Label>
            </div>
          </div>
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
                    <div className="flex items-center gap-2">
                      <Badge className={getSentimentBadgeClasses(item.sentiment)}>
                        {item.sentiment}
                      </Badge>
                      <Badge variant="outline" className="bg-slate-100">
                        {item.language === 'tl' ? 'Tagalog' : 'English'}
                      </Badge>
                    </div>
                  </div>

                  <div className="mt-2 flex justify-between text-xs text-slate-500">
                    <span>Confidence: {(item.confidence * 100).toFixed(1)}%</span>
                    <span>{item.timestamp.toLocaleTimeString()}</span>
                  </div>

                  {item.disasterType && item.disasterType !== "Not Specified" && (
                    <div className="mt-2 flex items-center gap-2">
                      <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                        {item.disasterType}
                      </Badge>
                      {item.location && (
                        <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                          {item.location}
                        </Badge>
                      )}
                    </div>
                  )}

                  {/* Only show explanation if it exists and is meaningful */}
                  {item.explanation && !item.explanation.includes("Fallback") && (
                    <div className="bg-slate-50 p-3 rounded-md border border-slate-200 mt-2">
                      <div className="flex items-start gap-2">
                        <AlertCircle className="h-5 w-5 text-slate-600 mt-0.5" />
                        <div>
                          <h4 className="text-sm font-medium mb-1">Analysis Details</h4>
                          <div className="space-y-2">
                            {item.disasterType && item.disasterType !== "Not Specified" && (
                              <p className="text-sm text-slate-700">
                                <span className="font-semibold">Disaster Type:</span> {item.disasterType}
                              </p>
                            )}
                            {item.location && (
                              <p className="text-sm text-slate-700">
                                <span className="font-semibold">Location:</span> {item.location}
                              </p>
                            )}
                            <p className="text-sm text-slate-700">
                              <span className="font-semibold">Analysis:</span> {item.explanation}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
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
                  `"${item.text}" - ${item.sentiment} (${(item.confidence * 100).toFixed(1)}%) - Language: ${item.language === 'tl' ? 'Tagalog' : 'English'}`
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