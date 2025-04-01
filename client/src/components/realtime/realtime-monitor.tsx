import { useState, useRef, useEffect } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { analyzeText } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { getSentimentBadgeClasses } from "@/lib/colors";
import { AlertCircle, Loader2 } from "lucide-react";
import { useDisasterContext } from "@/context/disaster-context";
import { motion } from "framer-motion";
import { SentimentFeedback } from "@/components/sentiment-feedback";

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

interface ProcessingStatus {
  processed: number;
  total: number;
  stage: string;
}

interface AnalysisProgress {
  isProcessing: boolean;
  startTime?: Date;
  status?: ProcessingStatus;
}

export function RealtimeMonitor() {
  const [text, setText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzedTexts, setAnalyzedTexts] = useState<AnalyzedText[]>([]);
  const [autoAnalyze, setAutoAnalyze] = useState(false);
  const [typingTimeout, setTypingTimeout] = useState<NodeJS.Timeout | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress>({
    isProcessing: false,
    status: {
      processed: 0,
      total: 0,
      stage: "Starting analysis..."
    }
  });
  const { toast } = useToast();
  const { refreshData } = useDisasterContext();

  // Auto-scroll to bottom of results
  const resultsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (resultsEndRef.current) {
      resultsEndRef.current.scrollIntoView({ behavior: "smooth" });
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

  // Update progress when receiving events
  useEffect(() => {
    const progressEventHandler = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'progress' && data.payload) {
          setAnalysisProgress(prev => ({
            ...prev,
            status: {
              processed: data.payload.processed || 0,
              total: data.payload.total || 0,
              stage: data.payload.stage || "Processing..."
            }
          }));
        }
      } catch (error) {
        console.error('Error parsing progress event:', error);
      }
    };

    window.addEventListener('message', progressEventHandler);
    return () => window.removeEventListener('message', progressEventHandler);
  }, []);

  const handleAnalyze = async () => {
    if (!text.trim()) {
      if (!autoAnalyze) {
        toast({
          title: "Empty text",
          description: "Please enter some text to analyze",
          variant: "destructive",
        });
      }
      return;
    }
    
    // YouTube-style performance tracking
    const startTime = performance.now();

    setIsAnalyzing(true);
    setAnalysisProgress({ 
      isProcessing: true, 
      startTime: new Date(),
      status: {
        processed: 0,
        total: 100,
        stage: "Starting analysis..."
      }
    });

    try {
      const normalizedText = text.trim().replace(/\s+/g, ' ');
      const hasFilipinoPhrases = /\b(ang|ng|mga|sa|ko|mo|nang|para|nung|yung|at|pag|ni|si|kay|na|po|opo|din|rin|nga|ba|eh|ay|ito|iyan|iyon|dito|diyan|doon)\b/i.test(normalizedText.toLowerCase());
      const result = await analyzeText(normalizedText);
      const detectedLanguage = hasFilipinoPhrases || result.post.language === 'tl' ? 'tl' : 'en';

      const analyzedText: AnalyzedText = {
        text: normalizedText,
        sentiment: result.post.sentiment,
        confidence: result.post.confidence,
        timestamp: new Date(),
        language: detectedLanguage,
        explanation: result.post.explanation,
        disasterType: result.post.disasterType,
        location: result.post.location
      };

      setAnalyzedTexts(prev => [...prev, analyzedText]);
      setText('');

      const isNonDisasterInput = !result.post.explanation || 
                              result.post.disasterType === "Not Specified" ||
                              result.post.disasterType === "UNKNOWN" ||
                              !result.post.disasterType;

      if (isNonDisasterInput && !autoAnalyze) {
        toast({
          title: 'Non-Disaster Input',
          description: 'This appears to be a non-disaster related input. For best results, please enter text about disaster situations.',
          variant: 'destructive',
          duration: 5000,
        });
      } else if (!autoAnalyze) {
        toast({
          title: 'Analysis complete',
          description: `Language: ${detectedLanguage === 'tl' ? 'Filipino' : 'English'}, Sentiment: ${result.post.sentiment}`,
        });
      }

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
      setAnalysisProgress({ isProcessing: false });
      
      // Log performance metrics for YouTube-like speed
      const endTime = performance.now();
      console.log(`Total analysis time: ${(endTime - startTime).toFixed(2)}ms`);
    }
  };

  const getProgressValue = () => {
    if (!analysisProgress.status) return 0;
    return (analysisProgress.status.processed / analysisProgress.status.total) * 100;
  };

  return (
    <div className="grid gap-4 md:grid-cols-2 h-full">
      {/* Input Card */}
      <Card className="border-none mb-2 sm:mb-4 overflow-hidden shadow-lg rounded-2xl bg-white/90 backdrop-blur-sm border border-indigo-100/40 flex flex-col h-full">
        <CardHeader className="p-4 border-b border-gray-200/40 bg-gradient-to-r from-indigo-600/90 via-blue-600/90 to-purple-600/90">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-full bg-white/20 backdrop-blur-sm shadow-inner">
                <AlertCircle className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
              </div>
              <div>
                <CardTitle className="text-base sm:text-lg font-bold text-white">
                  Real-Time Sentiment Analysis
                </CardTitle>
                <p className="text-xs sm:text-sm text-indigo-100 mt-0.5 sm:mt-1">
                  Enter text related to disaster situations
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 bg-white/20 backdrop-blur-sm px-3 py-1.5 rounded-lg">
              <Switch
                id="auto-analyze"
                checked={autoAnalyze}
                onCheckedChange={setAutoAnalyze}
              />
              <Label htmlFor="auto-analyze" className="text-white text-xs sm:text-sm">Auto Analyze</Label>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-5 flex flex-col h-full">
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter disaster-related text to analyze sentiment..."
            className="w-full flex-1 min-h-[150px] max-h-[250px] resize-y"
            style={{ height: "100%" }}
          />
        </CardContent>
        <CardFooter className="p-5 pt-0 flex flex-col gap-4">
          {isAnalyzing && (
            <motion.div 
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="w-full"
            >
              <div className="flex justify-between items-center mb-2">
                <div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                  <span className="text-sm font-medium text-slate-700">
                    {analysisProgress.status?.stage || "Analyzing..."}
                  </span>
                </div>
                {analysisProgress.startTime && (
                  <span className="text-xs text-slate-500">
                    {Math.round((new Date().getTime() - analysisProgress.startTime.getTime()) / 100) / 10}s
                  </span>
                )}
              </div>
              <Progress value={getProgressValue()} className="h-2" />
            </motion.div>
          )}
          <div className="flex justify-end w-full">
            <Button
              onClick={handleAnalyze}
              disabled={isAnalyzing || !text.trim()}
              className="bg-gradient-to-r from-indigo-600 via-blue-600 to-purple-600 hover:from-indigo-700 hover:via-blue-700 hover:to-purple-700 text-white shadow-lg hover:shadow-xl transition-all duration-300"
            >
              {isAnalyzing ? (
                <div className="flex items-center">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  <span>Processing...</span>
                </div>
              ) : (
                "Analyze Sentiment"
              )}
            </Button>
          </div>
        </CardFooter>
      </Card>

      {/* Results Card */}
      <Card className="border-none mb-2 sm:mb-4 overflow-hidden shadow-lg rounded-2xl bg-white/90 backdrop-blur-sm border border-indigo-100/40 flex flex-col h-full">
        <CardHeader className="p-4 border-b border-gray-200/40 bg-gradient-to-r from-indigo-600/90 via-blue-600/90 to-purple-600/90">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-full bg-white/20 backdrop-blur-sm shadow-inner">
              <Loader2 className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
            </div>
            <div>
              <CardTitle className="text-base sm:text-lg font-bold text-white">
                Analysis Results
              </CardTitle>
              <p className="text-xs sm:text-sm text-indigo-100 mt-0.5 sm:mt-1">
                {analyzedTexts.length === 0
                  ? "No results yet - analyze some text to see results"
                  : `Showing ${analyzedTexts.length} analyzed text${analyzedTexts.length !== 1 ? "s" : ""}`}
              </p>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-5 overflow-y-auto" style={{ height: "350px" }}>
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
            <div className="space-y-3">
              {analyzedTexts.map((item, index) => (
                <div key={index} className="p-3 bg-white/80 backdrop-blur-sm rounded-lg border border-slate-200/60 shadow-sm">
                  <div className="flex justify-between items-start">
                    <p className="text-sm text-slate-900 whitespace-pre-wrap break-words">
                      {item.text}
                    </p>
                    <div className="flex items-center gap-2">
                      <Badge className={getSentimentBadgeClasses(item.sentiment)}>
                        {item.sentiment}
                      </Badge>
                      <Badge variant="outline" className="bg-slate-100">
                        {item.language === "tl" ? "Filipino" : "English"}
                      </Badge>
                    </div>
                  </div>

                  <div className="mt-2 flex justify-between items-center text-xs text-slate-500">
                    <div className="flex items-center gap-2">
                      <span>Confidence: {(item.confidence * 100).toFixed(0)}%</span>
                      <SentimentFeedback 
                        originalText={item.text}
                        originalSentiment={item.sentiment}
                      />
                    </div>
                    <span>{item.timestamp.toLocaleTimeString()}</span>
                  </div>

                  {item.disasterType && item.disasterType !== "Not Specified" && item.disasterType !== "UNKNOWN" && (
                    <div className="mt-2 flex items-center gap-2">
                      <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                        {item.disasterType}
                      </Badge>
                      {item.location && item.location !== "UNKNOWN" && (
                        <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                          {item.location}
                        </Badge>
                      )}
                    </div>
                  )}

                  {/* Show explanation in quiz-like format if it exists and is meaningful */}
                  {item.explanation && !item.explanation.includes("Fallback") && (
                    <div className="bg-gradient-to-r from-blue-50/90 to-indigo-50/90 backdrop-blur-sm p-3 rounded-md border border-blue-200/50 mt-2 shadow-sm">
                      <div className="flex items-start gap-2">
                        <AlertCircle className="h-5 w-5 text-blue-600 mt-0.5" />
                        <div className="w-full">
                          <h4 className="text-sm font-medium mb-1 text-blue-800">AI Sentiment Quiz Results</h4>
                          
                          <div className="p-2 bg-white/80 rounded-md border border-blue-100 mb-2">
                            <p className="text-sm font-medium text-slate-700">Analyzing text: "{item.text}"</p>
                            <p className="text-xs text-slate-500 mt-1">What sentiment classification is most appropriate?</p>
                            
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-1 mt-2">
                              {["Panic", "Fear/Anxiety", "Disbelief", "Resilience", "Neutral"].map((sentiment, i) => (
                                <div key={i} className={`flex items-center px-2 py-1 rounded ${
                                  sentiment === item.sentiment 
                                    ? "bg-blue-100 border border-blue-300"
                                    : "bg-white border border-slate-200" 
                                }`}>
                                  <div className={`h-4 w-4 rounded-full flex items-center justify-center mr-2 ${
                                    sentiment === item.sentiment
                                      ? "bg-blue-600 text-white"
                                      : "bg-white border border-slate-300"
                                  }`}>
                                    {sentiment === item.sentiment && (
                                      <span className="text-xs text-white font-bold">✓</span>
                                    )}
                                  </div>
                                  <span className={`text-xs ${
                                    sentiment === item.sentiment
                                      ? "font-bold text-blue-800"
                                      : "text-slate-600"
                                  }`}>
                                    {String.fromCharCode(97 + i)}) {sentiment}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                          
                          <div className="space-y-2">
                            <div className="text-sm text-slate-700 p-2 bg-white/80 rounded border border-blue-100">
                              <span className="font-semibold text-blue-700">Explanation:</span>{" "}
                              <span className="text-slate-700">{item.explanation}</span>
                            </div>
                            
                            {/* Additional metadata in quiz-like cards */}
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2">
                              {item.disasterType && item.disasterType !== "Not Specified" && item.disasterType !== "UNKNOWN" && (
                                <div className="p-2 bg-white/80 rounded border border-blue-100">
                                  <span className="text-xs font-semibold text-blue-700">Disaster Type:</span>{" "}
                                  <span className="text-xs text-slate-700">{item.disasterType}</span>
                                </div>
                              )}
                              {item.location && item.location !== "UNKNOWN" && (
                                <div className="p-2 bg-white/80 rounded border border-blue-100">
                                  <span className="text-xs font-semibold text-blue-700">Location:</span>{" "}
                                  <span className="text-xs text-slate-700">{item.location}</span>
                                </div>
                              )}
                            </div>
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
              className="text-slate-600 hover:bg-red-50 hover:text-red-600 hover:border-red-200 font-medium"
            >
              Clear Results
            </Button>
            <Button
              variant="ghost"
              className="text-blue-600 hover:bg-blue-50 hover:text-blue-700 font-medium"
              onClick={() => {
                const text = analyzedTexts
                  .map(
                    (item) =>
                      `"${item.text}" - ${item.sentiment} (${(item.confidence * 100).toFixed(0)}%) - Language: ${item.language === "tl" ? "Filipino" : "English"}`,
                  )
                  .join("\n");
                navigator.clipboard.writeText(text);
                toast({
                  title: "Copied to clipboard",
                  description: "Analysis results have been copied to clipboard",
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