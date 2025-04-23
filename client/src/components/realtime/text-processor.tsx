import { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { processText, type TextProcessingResult } from '@/lib/api';
import { Zap, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

export function TextProcessor() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState<TextProcessingResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleProcess = useCallback(async () => {
    if (!inputText.trim()) {
      setError('Please enter text to process');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await processText(inputText);
      setResult(result);
    } catch (err) {
      console.error('Error processing text:', err);
      setError('Failed to process text. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [inputText]);

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { type: "spring", stiffness: 100 }
    }
  };

  return (
    <div className="flex flex-col space-y-6">
      <Card className="border-none overflow-hidden shadow-lg rounded-2xl bg-white/90 backdrop-blur-sm border border-indigo-100/40">
        <CardHeader className="p-4 bg-gradient-to-r from-indigo-600/90 via-blue-600/90 to-purple-600/90 border-b border-gray-200/40">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-full bg-white/20 backdrop-blur-sm shadow-inner">
              <Zap className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
            </div>
            <div>
              <CardTitle className="text-base sm:text-lg font-bold text-white">
                Text Processing
              </CardTitle>
              <CardDescription className="text-xs sm:text-sm text-indigo-100 mt-0.5">
                See how text is transformed through the NLP pipeline
              </CardDescription>
            </div>
          </div>
        </CardHeader>

        <CardContent className="p-6">
          <div className="space-y-4">
            <div>
              <label htmlFor="input-text" className="block text-sm font-medium text-gray-700 mb-1">
                Enter Text to Process
              </label>
              <Textarea
                id="input-text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter text to see how it's processed through normalization, tokenization, stemming, and final output..."
                className="min-h-[120px]"
              />
            </div>

            {error && (
              <div className="px-4 py-3 bg-red-50 border border-red-200 rounded-md">
                <p className="text-sm text-red-600">{error}</p>
              </div>
            )}

            <Button 
              onClick={handleProcess} 
              disabled={isLoading || !inputText.trim()} 
              className="w-full"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>Process Text</>
              )}
            </Button>

            {result && (
              <motion.div 
                initial="hidden"
                animate="visible"
                variants={containerVariants}
                className="mt-6 space-y-6"
              >
                <motion.div variants={itemVariants} className="space-y-2">
                  <h3 className="text-sm font-medium text-gray-700">Normalized Text</h3>
                  <div className="p-3 bg-gray-50 border border-gray-200 rounded-md text-sm overflow-auto max-h-32">
                    {result.normalizedText}
                  </div>
                </motion.div>

                <motion.div variants={itemVariants} className="space-y-2">
                  <h3 className="text-sm font-medium text-gray-700">Tokenized Text</h3>
                  <div className="p-3 bg-gray-50 border border-gray-200 rounded-md text-sm overflow-auto max-h-32">
                    <div className="flex flex-wrap gap-1.5">
                      {result.tokenizedText.map((token, index) => (
                        <span 
                          key={`${token}-${index}`}
                          className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                        >
                          {token}
                        </span>
                      ))}
                    </div>
                  </div>
                </motion.div>

                <motion.div variants={itemVariants} className="space-y-2">
                  <h3 className="text-sm font-medium text-gray-700">Stemmed Text</h3>
                  <div className="p-3 bg-gray-50 border border-gray-200 rounded-md text-sm overflow-auto max-h-32">
                    <div className="flex flex-wrap gap-1.5">
                      {result.stemmedText.map((stem, index) => (
                        <span 
                          key={`${stem}-${index}`}
                          className={cn(
                            "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium",
                            stem !== result.tokenizedText[index] 
                              ? "bg-purple-100 text-purple-800" 
                              : "bg-gray-100 text-gray-800"
                          )}
                        >
                          {stem}
                        </span>
                      ))}
                    </div>
                  </div>
                </motion.div>

                <motion.div variants={itemVariants} className="space-y-2">
                  <h3 className="text-sm font-medium text-gray-700">Final Cleaned Output</h3>
                  <div className="p-3 bg-gray-50 border border-gray-200 rounded-md text-sm overflow-auto max-h-32">
                    {result.finalOutput}
                  </div>
                </motion.div>
              </motion.div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}