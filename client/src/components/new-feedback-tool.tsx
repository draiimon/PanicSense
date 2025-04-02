import React, { useState, useEffect } from 'react';
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
  DialogClose,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { ThumbsUp, ThumbsDown, AlertCircle, Wand2, MapPin, AlertTriangle, CheckCircle2, RotateCw } from "lucide-react";

interface NewFeedbackToolProps {
  originalText: string;
  originalSentiment: string;
  originalLocation?: string;
  originalDisasterType?: string;
  onFeedbackSubmitted?: () => void;
}

// These are the preset options
const EMOTIONS = ["Panic", "Fear/Anxiety", "Disbelief", "Resilience", "Neutral"];
const DISASTER_TYPES = [
  "Fire", "Flood", "Earthquake", "Typhoon", "Tsunami", 
  "Landslide", "Volcanic Eruption", "Building Collapse",
  "Sunog", "Baha", "Lindol", "Bagyo", "Pagguho"
];
const LOCATIONS = [
  "Metro Manila", "Quezon City", "Manila", "Makati", 
  "Taguig", "Pasig", "Marikina", "Mandaluyong", 
  "Pasay", "Parañaque", "Cebu", "Davao", "TIP Manila"
];

export function NewFeedbackTool({
  originalText,
  originalSentiment,
  originalLocation = "Unknown",
  originalDisasterType = "Unknown",
  onFeedbackSubmitted
}: NewFeedbackToolProps) {
  // Main states
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("sentiment");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [resultDialogOpen, setResultDialogOpen] = useState(false);
  const [validationResult, setValidationResult] = useState<string | null>(null);
  const [processingStage, setProcessingStage] = useState<string | null>(null);
  
  // Form values
  const [correctedSentiment, setCorrectedSentiment] = useState<string>("");
  const [correctedLocation, setCorrectedLocation] = useState<string>("");
  const [correctedDisasterType, setCorrectedDisasterType] = useState<string>("");
  
  // Include checkboxes
  const [includeLocation, setIncludeLocation] = useState(false);
  const [includeDisasterType, setIncludeDisasterType] = useState(false);
  
  // Toast notifications
  const { toast } = useToast();

  // Reset form when dialog closes
  const resetForm = () => {
    setCorrectedSentiment("");
    setCorrectedLocation("");
    setCorrectedDisasterType("");
    setIncludeLocation(false);
    setIncludeDisasterType(false);
    setActiveTab("sentiment");
    setProcessingStage(null);
  };

  // Filter out the original sentiment to prevent selecting the same one
  const filteredSentiments = EMOTIONS.filter(
    emotion => emotion !== originalSentiment
  );

  // Handle form submission
  const handleSubmit = async () => {
    // Validation
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
        description: "Please enter a location or uncheck the location option",
        variant: "destructive",
      });
      setActiveTab("location");
      return;
    }

    if (includeDisasterType && !correctedDisasterType) {
      toast({
        title: "Disaster type required",
        description: "Please select a disaster type or uncheck the disaster type option",
        variant: "destructive",
      });
      setActiveTab("disaster");
      return;
    }

    // Start submission process
    setIsSubmitting(true);
    setProcessingStage("analyzing");
    
    try {
      // Artificial delay to show processing stages to the user
      await new Promise(resolve => setTimeout(resolve, 1200));
      setProcessingStage("validating");
      await new Promise(resolve => setTimeout(resolve, 800));
      setProcessingStage("applying");
      
      // Actually submit the feedback
      const response = await submitSentimentFeedback(
        originalText,
        originalSentiment,
        correctedSentiment || "",
        includeLocation ? correctedLocation : "",
        includeDisasterType ? correctedDisasterType : ""
      );
      
      console.log("Feedback response:", response);
      
      // Extract the message from the response
      const validationMessage = 
        response.status === "error" ? response.message : 
        response.message ? response.message :
        response.aiTrustMessage ? response.aiTrustMessage :
        "Feedback received. Thank you for helping improve our analysis.";
      
      // Set the validation result to show in the result dialog
      setValidationResult(validationMessage || null);
      
      // Dispatch a custom event for the data table to listen for
      const updateEvent = new CustomEvent('sentiment-data-changed', {
        detail: {
          text: originalText,
          newSentiment: correctedSentiment || undefined,
          newLocation: includeLocation ? correctedLocation : undefined,
          newDisasterType: includeDisasterType ? correctedDisasterType : undefined,
          timestamp: new Date().toISOString()
        }
      });
      window.dispatchEvent(updateEvent);
      
      // Close the dialog and show the result
      setIsOpen(false);
      setResultDialogOpen(true);
      
      // Call the callback if provided
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
      setProcessingStage(null);
    }
  };

  // Get sentiment badge color
  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'Panic': return 'bg-red-100 text-red-800 border-red-200';
      case 'Fear/Anxiety': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'Disbelief': return 'bg-purple-100 text-purple-800 border-purple-200';
      case 'Resilience': return 'bg-green-100 text-green-800 border-green-200';
      case 'Neutral': return 'bg-slate-100 text-slate-800 border-slate-200';
      default: return 'bg-slate-100 text-slate-800 border-slate-200';
    }
  };

  return (
    <>
      {/* Result Dialog - Shown after submission */}
      <AlertDialog open={resultDialogOpen} onOpenChange={setResultDialogOpen}>
        <AlertDialogContent className="max-w-md rounded-xl">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-blue-600 flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5" />
              Feedback Processed
            </AlertDialogTitle>
            <AlertDialogDescription className="text-base">
              <div className="p-4 border border-blue-200 bg-blue-50 rounded-md mb-3 whitespace-pre-line">
                {validationResult || "Your feedback has been received and applied to our system."}
              </div>
              <p className="text-sm text-gray-600 mt-2">
                Thank you for helping us improve the accuracy of our analysis!
              </p>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction className="bg-blue-600 hover:bg-blue-700 text-white rounded-full">
              Got it!
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Main Feedback Dialog */}
      <Dialog open={isOpen} onOpenChange={(open) => {
        setIsOpen(open);
        if (!open) resetForm();
      }}>
        <DialogTrigger asChild>
          <Button
            variant="outline" 
            size="sm"
            className="text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 rounded-full shadow-sm border border-slate-200"
            onClick={() => setIsOpen(true)}
          >
            <ThumbsDown className="h-4 w-4 mr-1.5" />
            <span className="text-xs font-medium">Provide Feedback</span>
          </Button>
        </DialogTrigger>
        
        <DialogContent className="sm:max-w-md rounded-xl">
          <DialogHeader>
            <DialogTitle className="text-xl text-indigo-700 flex items-center gap-2">
              <Wand2 className="h-5 w-5" />
              Improve Analysis Results
            </DialogTitle>
            <DialogDescription className="text-slate-600">
              Help us enhance our analysis by providing corrections.
            </DialogDescription>
          </DialogHeader>
          
          <div className="grid gap-4 py-3">
            {/* Original Text Section */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-slate-700">Original Text</h3>
              <div className="text-sm p-3 bg-slate-50 rounded-md border border-slate-200">
                {originalText}
              </div>
            </div>
            
            {/* Tab Navigation */}
            <Tabs defaultValue="sentiment" value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid grid-cols-3 mb-4 bg-slate-100 p-1 rounded-lg">
                <TabsTrigger 
                  value="sentiment"
                  className={activeTab === "sentiment" ? "bg-white shadow-sm" : "hover:bg-slate-50/80"}
                >
                  Sentiment
                </TabsTrigger>
                <TabsTrigger 
                  value="location"
                  className={activeTab === "location" ? "bg-white shadow-sm" : "hover:bg-slate-50/80"}
                >
                  Location
                </TabsTrigger>
                <TabsTrigger 
                  value="disaster"
                  className={activeTab === "disaster" ? "bg-white shadow-sm" : "hover:bg-slate-50/80"}
                >
                  Disaster Type
                </TabsTrigger>
              </TabsList>
              
              {/* Sentiment Correction Tab */}
              <TabsContent value="sentiment" className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h3 className="text-sm font-medium mb-2 text-slate-700">Current Classification</h3>
                    <div className="px-3 py-2 bg-slate-50 rounded-md border border-slate-200 flex items-center">
                      <Badge className={`${getSentimentColor(originalSentiment)} shadow-sm`}>
                        {originalSentiment}
                      </Badge>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-sm font-medium mb-2 text-slate-700">Suggest Correction</h3>
                    <Select value={correctedSentiment} onValueChange={setCorrectedSentiment}>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select sentiment" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectGroup>
                          <SelectLabel>Available Options</SelectLabel>
                          {filteredSentiments.map((sentiment) => (
                            <SelectItem key={sentiment} value={sentiment}>
                              <div className="flex items-center">
                                <Badge className={`mr-2 ${getSentimentColor(sentiment)}`}>
                                  {sentiment}
                                </Badge>
                              </div>
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </TabsContent>
              
              {/* Location Correction Tab */}
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
                    <Label 
                      htmlFor="include-location" 
                      className="text-sm font-medium text-slate-700"
                    >
                      Include location correction
                    </Label>
                  </div>
                  
                  {includeLocation && (
                    <div className="space-y-4 animate-in fade-in-50 duration-300">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <h3 className="text-sm font-medium mb-2 text-slate-700">Current Location</h3>
                          <div className="px-3 py-2 bg-slate-50 rounded-md border border-slate-200">
                            <div className="flex items-center">
                              <MapPin className="h-3.5 w-3.5 text-slate-400 mr-1.5" />
                              <span className="text-sm text-slate-700">
                                {originalLocation === "Unknown" ? "Not detected" : originalLocation}
                              </span>
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <h3 className="text-sm font-medium mb-2 text-slate-700">Suggest Location</h3>
                          <Select value={correctedLocation} onValueChange={setCorrectedLocation}>
                            <SelectTrigger className="w-full">
                              <SelectValue placeholder="Select location" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectGroup>
                                <SelectLabel>Common Locations</SelectLabel>
                                {LOCATIONS.map((location) => (
                                  <SelectItem key={location} value={location}>
                                    {location}
                                  </SelectItem>
                                ))}
                              </SelectGroup>
                            </SelectContent>
                          </Select>
                          <p className="text-xs text-slate-500 mt-2">
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
              
              {/* Disaster Type Correction Tab */}
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
                    <Label 
                      htmlFor="include-disaster" 
                      className="text-sm font-medium text-slate-700"
                    >
                      Include disaster type correction
                    </Label>
                  </div>
                  
                  {includeDisasterType && (
                    <div className="space-y-4 animate-in fade-in-50 duration-300">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <h3 className="text-sm font-medium mb-2 text-slate-700">Current Disaster Type</h3>
                          <div className="px-3 py-2 bg-slate-50 rounded-md border border-slate-200">
                            <div className="flex items-center">
                              <AlertTriangle className="h-3.5 w-3.5 text-slate-400 mr-1.5" />
                              <span className="text-sm text-slate-700">
                                {originalDisasterType === "Unknown" || originalDisasterType === "Not Specified" 
                                  ? "Not detected" 
                                  : originalDisasterType}
                              </span>
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <h3 className="text-sm font-medium mb-2 text-slate-700">Suggest Disaster Type</h3>
                          <Select value={correctedDisasterType} onValueChange={setCorrectedDisasterType}>
                            <SelectTrigger className="w-full">
                              <SelectValue placeholder="Select disaster type" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectGroup>
                                <SelectLabel>Disaster Types</SelectLabel>
                                {DISASTER_TYPES.map((type) => (
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
          
          {/* Processing stages animation */}
          {processingStage && (
            <div className="my-2 p-4 bg-blue-50 rounded-md border border-blue-100 animate-in fade-in-0 duration-300">
              <div className="flex items-center space-x-4">
                <div className="animate-spin">
                  <RotateCw className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-blue-700">
                    {processingStage === "analyzing" && "Analyzing your feedback..."}
                    {processingStage === "validating" && "Validating correction accuracy..."}
                    {processingStage === "applying" && "Applying changes to database..."}
                  </p>
                </div>
              </div>
            </div>
          )}
          
          <DialogFooter className="flex justify-between items-center gap-4 sm:gap-0">
            <div className="text-xs text-slate-500 italic">
              All feedback helps improve our system
            </div>
            <div className="flex gap-2">
              <DialogClose asChild>
                <Button variant="outline" className="rounded-full">Cancel</Button>
              </DialogClose>
              <Button 
                onClick={handleSubmit} 
                className="rounded-full bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 text-white"
                disabled={isSubmitting || (!correctedSentiment && !includeLocation && !includeDisasterType)}
              >
                {isSubmitting ? (
                  <>
                    <RotateCw className="h-4 w-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  "Submit Feedback"
                )}
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}