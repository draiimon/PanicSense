import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { submitSentimentFeedback, SentimentFeedback as SentimentFeedbackType } from "@/lib/api";
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
  const [warningOpen, setWarningOpen] = useState(false);
  const [warningMessage, setWarningMessage] = useState("");
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
  
  // General purpose client-side validation function for all sentiment changes
  const validateSentimentChange = (): { valid: boolean, message: string | null } => {
    // Default to valid
    let result = { valid: true, message: null };
    
    // ===== BASIC CONTENT CLASSIFICATION =====
    const hasJokeIndicators = containsJokeIndicators(originalText);
    const hasDisasterKeywords = containsDisasterKeywords(originalText);
    const hasQuestionForm = containsQuestionForm(originalText);
    const hasEmojis = originalText.includes("😂") || originalText.includes("🤣") || originalText.includes("😆");
    
    // ===== SENTIMENT CLASSIFICATION RULES =====
    
    // RULE 1: Joking text should be classified as Disbelief
    if (correctedSentiment !== "Disbelief" && hasJokeIndicators && !hasSeriosDisasterIndication(originalText)) {
      if (hasJokeIndicators && (hasDisasterKeywords || hasEmojis)) {
        return {
          valid: false,
          message: "This text appears to be joking or sarcastic. It should be classified as Disbelief, not " + correctedSentiment
        };
      }
    }
    
    // RULE 2: Clear disaster text without humor should not be classified as Disbelief
    if (correctedSentiment === "Disbelief" && hasDisasterKeywords && !hasJokeIndicators && !hasQuestionForm) {
      return {
        valid: false,
        message: "This text appears to contain serious disaster content without humor indicators. It should not be classified as Disbelief."
      };
    }
    
    // RULE 3: Text with both disaster keywords AND joke indicators is valid as Disbelief
    if (correctedSentiment === "Disbelief" && hasJokeIndicators && hasDisasterKeywords) {
      // Allow the change - if someone is joking about a disaster, it's valid to mark as Disbelief
      return { valid: true, message: null };
    }
    
    // RULE 4: Text with emojis and question markers should be allowed to be Disbelief 
    if (correctedSentiment === "Disbelief" && hasEmojis && hasQuestionForm) {
      // Specifically for the "MA SUNOG DAW?" text with laughing emojis
      return { valid: true, message: null };
    }
    
    return result;
  };
  
  // Helper function to check if text is in question form (daw/raw/?, etc)
  const containsQuestionForm = (text: string): boolean => {
    const lowerText = text.toLowerCase();
    return (
      text.includes("?") || 
      lowerText.includes("daw") || 
      lowerText.includes("raw") || 
      lowerText.includes("ba") || 
      lowerText.includes("kaya") || 
      lowerText.includes("talaga")
    );
  };
  
  // Helper function to detect serious disaster indications that override joke markers
  const hasSeriosDisasterIndication = (text: string): boolean => {
    const lowerText = text.toLowerCase();
    const seriousIndicators = [
      "mamatay", "patay", "papatayin", "namatay",
      "died", "dead", "death", "killed",
      "casualties", "casualty", "victim", "victims",
      "injured", "injuries", "wounded", "wound",
      "emergency", "emerhensi", "evac", "evacuate"
    ];
    
    return seriousIndicators.some(indicator => lowerText.includes(indicator));
  };
  
  // Function to check if text contains joke/sarcasm indicators
  const containsJokeIndicators = (text: string): boolean => {
    const lowerText = text.toLowerCase();
    const jokeIndicators = [
      "haha", "hehe", "lol", "lmao", "ulol", "gago", "tanga", 
      "daw?", "raw?", "talaga?", "really?", "😂", "🤣",
      "joke", "jokes", "joke lang", "eme", "charot", "char", "joke time",
      "jk", "kidding", "just kidding", "sarcasm"
    ];
    
    // Check for laughter patterns
    if (jokeIndicators.some(indicator => lowerText.includes(indicator))) {
      return true;
    }
    
    // Check for multiple exclamation marks with "haha"
    if (lowerText.includes("haha") && text.includes("!")) {
      return true;
    }
    
    // Check for capitalized laughter
    if (text.includes("HAHA") || text.includes("HEHE")) {
      return true;
    }
    
    return false;
  };
  
  // Function to check if text contains disaster keywords
  const containsDisasterKeywords = (text: string): boolean => {
    const lowerText = text.toLowerCase();
    const disasterKeywords = [
      "earthquake", "lindol", "fire", "sunog", "flood", "baha", 
      "typhoon", "bagyo", "tsunami", "landslide", "nagiba"
    ];
    
    return disasterKeywords.some(keyword => lowerText.includes(keyword));
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
    
    // Check if sentiment change is valid (only if correctedSentiment is provided)
    if (correctedSentiment) {
      const validationResult = validateSentimentChange();
      if (!validationResult.valid && validationResult.message) {
        // Show validation warning directly in UI
        setWarningMessage(validationResult.message);
        setWarningOpen(true);
        // Return without submitting - this blocks the submission completely
        return;
      }
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
      const response = await submitSentimentFeedback(
        originalText,
        originalSentiment,
        correctedSentiment,
        includeLocation ? correctedLocation : undefined,
        includeDisasterType ? correctedDisasterType : undefined
      );

      console.log("Raw response from server:", JSON.stringify(response));
      console.log("Successfully parsed sentiment feedback response:", response);

      // Check if response contains a warning flag for trolling detection
      if (response.possibleTrolling && response.aiTrustMessage) {
        // Show warning popup with AI-generated message
        setWarningMessage(response.aiTrustMessage);
        setWarningOpen(true);
      } else {
        // Show success message
        toast({
          title: "Feedback submitted",
          description: "Thank you for helping improve our analysis system",
        });
      }
      
      setIsOpen(false);
      resetForm();
      
      // Always call the onFeedbackSubmitted callback to force UI refresh immediately
      // regardless of warning or success, making sure frontend updates instantly
      if (onFeedbackSubmitted) {
        onFeedbackSubmitted();
      }
    } catch (error) {
      console.error("Error submitting sentiment feedback:", error);
      toast({
        title: "Submission failed",
        description: "There was an error submitting your feedback",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  // We removed the WebSocket connection as it was causing issues
  // Instead, we're directly showing warnings based on API responses

  return (
    <>
      <AlertDialog open={warningOpen} onOpenChange={setWarningOpen}>
        <AlertDialogContent className="max-w-md">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-red-600 flex items-center">
              <AlertCircle className="mr-2 h-5 w-5" />
              Potential Feedback Issue Detected
            </AlertDialogTitle>
            <AlertDialogDescription className="text-base">
              <div className="p-3 border border-red-200 bg-red-50 rounded-md mb-3">
                {warningMessage || "Our AI detected an inconsistency in your feedback."}
              </div>
              <p className="text-sm text-gray-600 mt-2">
                {correctedSentiment && validateSentimentChange().valid === false ? (
                  // Client-side validation message (submission blocked)
                  "Your feedback cannot be submitted due to this validation issue. Please review your changes."
                ) : (
                  // Server-side warning message (submission allowed)
                  "Your feedback has been saved and all changes have been applied to the database. This alert is just to inform you about a potential mismatch between the text content and your chosen category."
                )}
              </p>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction className="bg-red-600 hover:bg-red-700 text-white">
              {correctedSentiment && validateSentimentChange().valid === false ? 
                "Go Back and Fix" : 
                "I Understand"
              }
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      
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
                  <Select 
                    value={correctedSentiment} 
                    onValueChange={setCorrectedSentiment}
                  >
                    <SelectTrigger className={`w-full ${correctedSentiment && !validateSentimentChange().valid ? "border-red-500" : ""}`}>
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
                  
                  {/* Show instant validation feedback */}
                  {correctedSentiment && !validateSentimentChange().valid && (
                    <div className="text-sm text-red-600 mt-1 flex items-center">
                      <AlertCircle className="h-3.5 w-3.5 mr-1" />
                      {validateSentimentChange().message}
                    </div>
                  )}
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
            disabled={
              isSubmitting || 
              (!correctedSentiment && !includeLocation && !includeDisasterType) ||
              (correctedSentiment && !validateSentimentChange().valid)
            }
            className={`
              bg-gradient-to-r 
              ${correctedSentiment && !validateSentimentChange().valid 
                ? "from-red-500 to-red-700 opacity-70" 
                : "from-indigo-600 to-purple-600"
              }
            `}
          >
            {isSubmitting ? "Submitting..." : "Submit Feedback"}
            {!isSubmitting && <ThumbsUp className="ml-2 h-4 w-4" />}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
    </>
  );
}