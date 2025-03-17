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

interface AnalysisProgress {
  step: number;
  message: string;
  progress: number;
}

export function RealtimeMonitor() {
  const [text, setText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzedTexts, setAnalyzedTexts] = useState<AnalyzedText[]>([]);
  const [autoAnalyze, setAutoAnalyze] = useState(false);
  const [typingTimeout, setTypingTimeout] = useState<NodeJS.Timeout | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress>({
    step: 0,
    message: "",
    progress: 0
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

  // Function to normalize text while preserving special characters
  const normalizeText = (input: string): string => {
    return input
      .replace(/[\u200B-\u200D\uFEFF]/g, '') // Remove zero-width characters
      .replace(/\r\n/g, '\n') // Normalize line endings
      .replace(/\n+/g, ' ') // Replace multiple newlines with space
      .replace(/\s+/g, ' ') // Replace multiple spaces with single space
      .trim(); // Remove leading/trailing whitespace
  };

  // Function to update progress during analysis
  const updateProgress = (step: number, message: string, progress: number) => {
    setAnalysisProgress({
      step,
      message,
      progress
    });
  };

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

    setIsAnalyzing(true);
    try {
      // Step 1: Text Normalization
      updateProgress(1, "Normalizing text...", 20);
      const normalizedText = normalizeText(text);

      // Step 2: Language Detection
      updateProgress(2, "Detecting language...", 40);
      const hasFilipinoPhrases = /\b(ang|ng|mga|sa|ko|mo|nang|para|nung|yung|at|pag|ni|si|kay|na|po|opo|din|rin|nga|ba|eh|ay|ito|iyan|iyon|dito|diyan|doon)\b/i.test(normalizedText.toLowerCase());

      // Step 3: Sentiment Analysis
      updateProgress(3, "Analyzing sentiment...", 60);
      const result = await analyzeText(normalizedText);

      // Step 4: Processing Results
      updateProgress(4, "Processing results...", 80);
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

      // Step 5: Finalizing
      updateProgress(5, "Finalizing analysis...", 100);
      setAnalyzedTexts(prev => [...prev, analyzedText]);
      setText('');

      // Only check for disaster-related content
      const isNonDisasterInput = !result.post.explanation || 
                              result.post.disasterType === "Not Specified" ||
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
      // Reset progress after completion or error
      setTimeout(() => {
        setIsAnalyzing(false);
        setAnalysisProgress({
          step: 0,
          message: "",
          progress: 0
        });
      }, 500);
    }
  };

  return (
    <div className="grid gap-6 md:grid-cols-2">
      {/* Input Card */}
      <Card className="bg-white rounded-lg shadow">
        <CardHeader className="p-5 border-b border-gray-200">
          <div className="flex justify-between items-center">
            <div>
              <CardTitle className="text-lg font-medium text-slate-800">
                Real-Time Sentiment Analysis
              </CardTitle>
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
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter disaster-related text to analyze sentiment..."
            className="min-h-[200px]"
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
                <span className="text-sm font-medium text-slate-700">
                  {analysisProgress.message}
                </span>
                <span className="text-sm text-slate-500">
                  {analysisProgress.progress}%
                </span>
              </div>
              <Progress value={analysisProgress.progress} className="h-2" />
            </motion.div>
          )}
          <div className="flex justify-end w-full">
            <Button
              onClick={handleAnalyze}
              disabled={isAnalyzing || !text.trim()}
              className="bg-blue-600 hover:bg-blue-700"
            >
              {isAnalyzing ? (
                <div className="flex items-center">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  <span>Analyzing...</span>
                </div>
              ) : (
                "Analyze Sentiment"
              )}
            </Button>
          </div>
        </CardFooter>
      </Card>

      {/* Results Card */}
      <Card className="bg-white rounded-lg shadow">
        <CardHeader className="p-5 border-b border-gray-200">
          <CardTitle className="text-lg font-medium text-slate-800">
            Analysis Results
          </CardTitle>
          <CardDescription className="text-sm text-slate-500">
            {analyzedTexts.length === 0
              ? "No results yet - analyze some text to see results"
              : `Showing ${analyzedTexts.length} analyzed text${analyzedTexts.length !== 1 ? "s" : ""}`}
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
                                <span className="font-semibold">Disaster Type:</span>{" "}
                                {item.disasterType}
                              </p>
                            )}
                            {item.location && (
                              <p className="text-sm text-slate-700">
                                <span className="font-semibold">Location:</span>{" "}
                                {item.location}
                              </p>
                            )}
                            <p className="text-sm text-slate-700">
                              <span className="font-semibold">Analysis:</span>{" "}
                              {item.explanation}
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
                const text = analyzedTexts
                  .map(
                    (item) =>
                      `"${item.text}" - ${item.sentiment} (${(item.confidence * 100).toFixed(1)}%) - Language: ${item.language === "tl" ? "Filipino" : "English"}`,
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