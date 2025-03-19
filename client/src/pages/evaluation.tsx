import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useDisasterContext } from "@/context/disaster-context";
import { getAnalyzedFile, getSentimentPostsByFileId, getSentimentPosts } from "@/lib/api";
import { ConfusionMatrix } from "@/components/evaluation/confusion-matrix";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FileUploader } from "@/components/file-uploader";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { DatabaseIcon, FileTextIcon } from "lucide-react";

export default function Evaluation() {
  const { analyzedFiles, isLoadingAnalyzedFiles, sentimentPosts: allSentimentPosts } = useDisasterContext();
  const [selectedFileId, setSelectedFileId] = useState<string>("all");
  const [totalRecords, setTotalRecords] = useState<number>(0);

  // Fetch metrics for selected file
  const { 
    data: selectedFile,
    isLoading: isLoadingSelectedFile 
  } = useQuery({
    queryKey: ['/api/analyzed-files', selectedFileId],
    queryFn: () => getAnalyzedFile(parseInt(selectedFileId)),
    enabled: !!selectedFileId && selectedFileId !== "all"
  });

  // Fetch sentiment posts for selected file
  const { 
    data: sentimentPosts,
    isLoading: isLoadingSentimentPosts 
  } = useQuery({
    queryKey: ['/api/sentiment-posts/file', selectedFileId],
    queryFn: () => getSentimentPostsByFileId(parseInt(selectedFileId)),
    enabled: !!selectedFileId && selectedFileId !== "all"
  });

  // Fetch all sentiment posts if "All Datasets" is selected 
  // Using the correct query key to match the one in the API
  const {
    data: allData,
    isLoading: isLoadingAllData
  } = useQuery({
    queryKey: ['/api/sentiment-posts'],
    queryFn: () => getSentimentPosts(),
    enabled: selectedFileId === "all"
  });

  const isLoading = 
    isLoadingAnalyzedFiles || 
    isLoadingSelectedFile || 
    isLoadingSentimentPosts || 
    (selectedFileId === "all" && isLoadingAllData);

  const getDisplayData = () => {
    if (selectedFileId === "all") {
      return {
        posts: allData || allSentimentPosts || [],
        name: "All Datasets",
        isAll: true
      };
    }

    return {
      posts: sentimentPosts || [],
      name: selectedFile?.originalName || "",
      isAll: false
    };
  };

  const { posts, name, isAll } = getDisplayData();

  return (
    <div className="space-y-6">
      {/* Evaluation Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Evaluation Metrics</h1>
          <p className="mt-1 text-sm text-slate-500">Assess the accuracy of sentiment analysis predictions</p>
        </div>
        <FileUploader 
          className="mt-4 sm:mt-0"
          onSuccess={(data) => {
            if (data.file?.id) {
              setSelectedFileId(data.file.id.toString());
            }
          }}
        />
      </div>

      {/* File Selection */}
      <Card className="bg-white rounded-lg shadow">
        <CardHeader className="px-6 py-5 border-b border-gray-200">
          <CardTitle className="text-lg font-medium text-slate-800">Select Dataset</CardTitle>
          <CardDescription className="text-sm text-slate-500">
            Choose a dataset to view its evaluation metrics or select "All Datasets" for combined analysis
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          {isLoadingAnalyzedFiles ? (
            <div className="flex items-center justify-center h-20">
              <svg className="animate-spin h-6 w-6 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            </div>
          ) : analyzedFiles.length === 0 ? (
            <div className="text-center py-6 border border-dashed border-slate-300 rounded-lg">
              <DatabaseIcon className="h-10 w-10 text-slate-400 mx-auto mb-2" />
              <p className="text-slate-600 font-medium">No analyzed files available</p>
              <p className="text-sm text-slate-500 mt-1">Upload a CSV file to generate evaluation metrics</p>
              <Button variant="outline" className="mt-4" onClick={() => document.getElementById('file-upload')?.click()}>
                <FileTextIcon className="h-4 w-4 mr-2" />
                Upload CSV File
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              <Select
                value={selectedFileId}
                onValueChange={setSelectedFileId}
              >
                <SelectTrigger className="w-full bg-white">
                  <SelectValue placeholder="Select a file" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all" className="font-medium text-blue-600">
                    <div className="flex items-center">
                      <DatabaseIcon className="h-4 w-4 mr-2" />
                      All Datasets ({allSentimentPosts?.length || 0} total records)
                    </div>
                  </SelectItem>
                  <div className="py-1 px-2 text-xs text-slate-500 border-b">Individual Datasets</div>
                  {analyzedFiles.map((file) => (
                    <SelectItem key={file.id} value={file.id.toString()}>
                      {file.originalName} ({file.recordCount} records)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </CardContent>
      </Card>

      {selectedFileId && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <ConfusionMatrix 
            fileId={selectedFileId !== "all" ? parseInt(selectedFileId) : undefined}
            confusionMatrix={selectedFile?.evaluationMetrics?.confusionMatrix}
            title={isAll ? "Sentiment Analysis Performance" : "Sentiment Analysis Performance"}
            description={isAll 
              ? "Detailed model prediction accuracy and metrics across all datasets" 
              : `Detailed model prediction accuracy and metrics for ${name}`
            }
            allDatasets={selectedFileId === "all"}
          />
        </motion.div>
      )}
    </div>
  );
}