import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { submitSentimentFeedback } from "@/lib/api";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ThumbsUp, ThumbsDown } from "lucide-react";

interface SentimentFeedbackProps {
  originalText: string;
  originalSentiment: string;
  onFeedbackSubmitted?: () => void;
}

export function SentimentFeedback({ originalText, originalSentiment, onFeedbackSubmitted }: SentimentFeedbackProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [correctedSentiment, setCorrectedSentiment] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { toast } = useToast();

  const sentimentOptions = [
    "Panic",
    "Fear/Anxiety",
    "Disbelief",
    "Resilience",
    "Neutral"
  ];

  // Filter out the original sentiment from options
  const filteredOptions = sentimentOptions.filter(
    sentiment => sentiment !== originalSentiment
  );

  const handleSubmit = async () => {
    if (!correctedSentiment) {
      toast({
        title: "Selection required",
        description: "Please select a corrected sentiment",
        variant: "destructive",
      });
      return;
    }

    setIsSubmitting(true);
    try {
      await submitSentimentFeedback(
        originalText,
        originalSentiment,
        correctedSentiment
      );

      toast({
        title: "Feedback submitted",
        description: "Thank you for helping improve our sentiment analysis",
      });
      setIsOpen(false);
      setCorrectedSentiment("");
      
      // Call the onFeedbackSubmitted callback if provided
      if (onFeedbackSubmitted) {
        onFeedbackSubmitted();
      }
    } catch (error) {
      console.error("Error submitting feedback:", error);
      toast({
        title: "Submission failed",
        description: "There was an error submitting your feedback",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button
          variant="ghost" 
          size="sm"
          className="text-slate-500 hover:text-indigo-600 hover:bg-indigo-50"
          onClick={() => setIsOpen(true)}
        >
          <ThumbsDown className="h-4 w-4 mr-1" />
          <span className="text-xs">Incorrect?</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Improve Sentiment Analysis</DialogTitle>
          <DialogDescription>
            Your feedback helps us make our sentiment analysis more accurate.
          </DialogDescription>
        </DialogHeader>
        
        <div className="grid gap-4 py-4">
          <div className="space-y-2">
            <h3 className="text-sm font-medium">Original Text</h3>
            <p className="text-sm p-2 bg-slate-50 rounded border border-slate-200">
              {originalText}
            </p>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h3 className="text-sm font-medium mb-2">Current Analysis</h3>
              <div className="px-3 py-2 bg-red-50 text-red-700 rounded border border-red-200">
                {originalSentiment}
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-medium mb-2">Corrected Sentiment</h3>
              <Select value={correctedSentiment} onValueChange={setCorrectedSentiment}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select sentiment" />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectLabel>Available Sentiments</SelectLabel>
                    {filteredOptions.map((sentiment) => (
                      <SelectItem key={sentiment} value={sentiment}>
                        {sentiment}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
        
        <DialogFooter className="sm:justify-between">
          <Button
            variant="outline"
            onClick={() => {
              setIsOpen(false);
              setCorrectedSentiment("");
            }}
          >
            Cancel
          </Button>
          <Button 
            onClick={handleSubmit} 
            disabled={isSubmitting || !correctedSentiment}
            className="bg-gradient-to-r from-indigo-600 to-purple-600"
          >
            {isSubmitting ? "Submitting..." : "Submit Feedback"}
            {!isSubmitting && <ThumbsUp className="ml-2 h-4 w-4" />}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}