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
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ThumbsUp, ThumbsDown, MapPin, AlertCircle } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";

interface SentimentFeedbackProps {
  originalText: string;
  originalSentiment: string;
  originalLocation?: string;
  originalDisasterType?: string;
  onFeedbackSubmitted?: () => void;
}

export function SentimentFeedback({ 
  originalText, 
  originalSentiment, 
  originalLocation = "UNKNOWN", 
  originalDisasterType = "UNKNOWN", 
  onFeedbackSubmitted 
}: SentimentFeedbackProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [correctedSentiment, setCorrectedSentiment] = useState<string>("");
  const [correctedLocation, setCorrectedLocation] = useState<string>("");
  const [correctedDisasterType, setCorrectedDisasterType] = useState<string>("");
  const [activeTab, setActiveTab] = useState<string>("sentiment");
  const [includeLocation, setIncludeLocation] = useState<boolean>(false);
  const [includeDisasterType, setIncludeDisasterType] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { toast } = useToast();

  const sentimentOptions = [
    "Panic",
    "Fear/Anxiety",
    "Disbelief",
    "Resilience",
    "Neutral"
  ];

  const disasterTypeOptions = [
    "Typhoon",
    "Flood",
    "Earthquake",
    "Landslide",
    "Volcanic Eruption",
    "Tsunami",
    "Fire",
    "Drought",
    "Storm Surge",
    "Other"
  ];

  // Common Philippine locations
  const locationSuggestions = [
    "Manila", "Quezon City", "Davao", "Cebu", "Baguio",
    "Tacloban", "Batangas", "Cavite", "Laguna", "Rizal",
    "Pampanga", "Bulacan", "Zambales", "Bataan", "Ilocos",
    "Pangasinan", "La Union", "Isabela", "Cagayan", "Bicol",
    "Sorsogon", "Albay", "Camarines Sur", "Camarines Norte", "Palawan",
    "Mindoro", "Marinduque", "Romblon", "Aklan", "Antique",
    "Capiz", "Iloilo", "Negros", "Leyte", "Samar",
    "Bohol", "Bukidnon", "Misamis", "Zamboanga", "Basilan",
    "Sulu", "Tawi-Tawi", "Cotabato", "Maguindanao", "Sultan Kudarat",
    "South Cotabato", "Agusan", "Surigao", "Dinagat Islands"
  ];

  // Filter out the original sentiment from options
  const filteredOptions = sentimentOptions.filter(
    sentiment => sentiment !== originalSentiment
  );

  const resetForm = () => {
    setCorrectedSentiment("");
    setCorrectedLocation("");
    setCorrectedDisasterType("");
    setIncludeLocation(false);
    setIncludeDisasterType(false);
    setActiveTab("sentiment");
  };

  const handleSubmit = async () => {
    if (!correctedSentiment && !includeLocation && !includeDisasterType) {
      toast({
        title: "Selection required",
        description: "Please select at least one correction to provide feedback on",
        variant: "destructive",
      });
      return;
    }

    if (includeLocation && !correctedLocation) {
      toast({
        title: "Location required",
        description: "Please enter a location or uncheck the location checkbox",
        variant: "destructive",
      });
      setActiveTab("location");
      return;
    }

    if (includeDisasterType && !correctedDisasterType) {
      toast({
        title: "Disaster type required",
        description: "Please select a disaster type or uncheck the disaster type checkbox",
        variant: "destructive",
      });
      setActiveTab("disaster");
      return;
    }

    setIsSubmitting(true);
    try {
      await submitSentimentFeedback(
        originalText,
        originalSentiment,
        correctedSentiment,
        includeLocation ? correctedLocation : undefined,
        includeDisasterType ? correctedDisasterType : undefined
      );

      toast({
        title: "Feedback submitted",
        description: "Thank you for helping improve our analysis system",
      });
      setIsOpen(false);
      resetForm();
      
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
          
          <Tabs defaultValue="sentiment" value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid grid-cols-3 mb-4">
              <TabsTrigger value="sentiment">Sentiment</TabsTrigger>
              <TabsTrigger value="location">Location</TabsTrigger>
              <TabsTrigger value="disaster">Disaster Type</TabsTrigger>
            </TabsList>
            
            <TabsContent value="sentiment" className="space-y-4">
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
            </TabsContent>
            
            <TabsContent value="location" className="space-y-4">
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="include-location" 
                    checked={includeLocation}
                    onCheckedChange={(checked) => {
                      if (typeof checked === 'boolean') setIncludeLocation(checked);
                      if (!checked) setCorrectedLocation("");
                    }}
                  />
                  <label 
                    htmlFor="include-location" 
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    Include location correction
                  </label>
                </div>
                
                {includeLocation && (
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <h3 className="text-sm font-medium mb-2">Current Location</h3>
                        <div className="px-3 py-2 bg-blue-50 text-blue-700 rounded border border-blue-200">
                          {originalLocation === "UNKNOWN" ? "Not detected" : originalLocation}
                        </div>
                      </div>
                      
                      <div>
                        <h3 className="text-sm font-medium mb-2">Corrected Location</h3>
                        <Select value={correctedLocation} onValueChange={setCorrectedLocation}>
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select location" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectGroup>
                              <SelectLabel>Common Locations</SelectLabel>
                              {locationSuggestions.map((location) => (
                                <SelectItem key={location} value={location}>
                                  {location}
                                </SelectItem>
                              ))}
                            </SelectGroup>
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-gray-500 mt-1">
                          Or type a custom location:
                        </p>
                        <Input 
                          placeholder="Enter location" 
                          value={correctedLocation}
                          onChange={(e) => setCorrectedLocation(e.target.value)}
                          className="mt-1"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </TabsContent>
            
            <TabsContent value="disaster" className="space-y-4">
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="include-disaster" 
                    checked={includeDisasterType}
                    onCheckedChange={(checked) => {
                      if (typeof checked === 'boolean') setIncludeDisasterType(checked);
                      if (!checked) setCorrectedDisasterType("");
                    }}
                  />
                  <label 
                    htmlFor="include-disaster" 
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    Include disaster type correction
                  </label>
                </div>
                
                {includeDisasterType && (
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <h3 className="text-sm font-medium mb-2">Current Disaster Type</h3>
                        <div className="px-3 py-2 bg-amber-50 text-amber-700 rounded border border-amber-200">
                          {originalDisasterType === "UNKNOWN" ? "Not detected" : originalDisasterType}
                        </div>
                      </div>
                      
                      <div>
                        <h3 className="text-sm font-medium mb-2">Corrected Disaster Type</h3>
                        <Select value={correctedDisasterType} onValueChange={setCorrectedDisasterType}>
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select disaster type" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectGroup>
                              <SelectLabel>Disaster Types</SelectLabel>
                              {disasterTypeOptions.map((type) => (
                                <SelectItem key={type} value={type}>
                                  {type}
                                </SelectItem>
                              ))}
                            </SelectGroup>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </div>
        
        <DialogFooter className="sm:justify-between">
          <Button
            variant="outline"
            onClick={() => {
              setIsOpen(false);
              resetForm();
            }}
          >
            Cancel
          </Button>
          <Button 
            onClick={handleSubmit} 
            disabled={isSubmitting || (!correctedSentiment && !includeLocation && !includeDisasterType)}
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