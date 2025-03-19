import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { useDisasterContext } from "@/context/disaster-context";
import { getAnalyzedFile, getSentimentPostsByFileId, getSentimentPosts } from "@/lib/api";
import { MetricsDisplay } from "@/components/evaluation/metrics-display";
import { ConfusionMatrix } from "@/components/evaluation/confusion-matrix";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FileUploader } from "@/components/file-uploader";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { DatabaseIcon, BarChart3Icon, FileTextIcon } from "lucide-react";

export default function Evaluation() {
  const { analyzedFiles, isLoadingAnalyzedFiles, sentimentPosts: allSentimentPosts } = useDisasterContext();
  const [selectedFileId, setSelectedFileId] = useState<string>("all");
  const [activeTab, setActiveTab] = useState<string>("confusion");
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
  const {
    data: allData,
    isLoading: isLoadingAllData
  } = useQuery({
    queryKey: ['/api/sentiment-posts/all'],
    queryFn: () => getSentimentPosts(),
    enabled: selectedFileId === "all"
  });

  // Calculate total records analyzed
  useEffect(() => {
    if (selectedFileId === "all") {
      setTotalRecords(allSentimentPosts?.length || 0);
    } else if (selectedFile) {
      setTotalRecords(selectedFile.recordCount);
    }
  }, [selectedFileId, selectedFile, allSentimentPosts]);

  const isLoading = 
    isLoadingAnalyzedFiles || 
    isLoadingSelectedFile || 
    isLoadingSentimentPosts || 
    (selectedFileId === "all" && isLoadingAllData);

  // Calculate average metrics across all files - for the "All Datasets" option
  const calculateAverageMetrics = () => {
    if (!analyzedFiles || analyzedFiles.length === 0) return undefined;
    
    const filesWithMetrics = analyzedFiles.filter(file => file.evaluationMetrics);
    if (filesWithMetrics.length === 0) return undefined;
    
    const sum = filesWithMetrics.reduce((acc, file) => {
      const metrics = file.evaluationMetrics;
      if (!metrics) return acc;
      
      return {
        accuracy: acc.accuracy + metrics.accuracy,
        precision: acc.precision + metrics.precision,
        recall: acc.recall + metrics.recall,
        f1Score: acc.f1Score + metrics.f1Score
      };
    }, { accuracy: 0, precision: 0, recall: 0, f1Score: 0 });
    
    return {
      accuracy: sum.accuracy / filesWithMetrics.length,
      precision: sum.precision / filesWithMetrics.length,
      recall: sum.recall / filesWithMetrics.length,
      f1Score: sum.f1Score / filesWithMetrics.length
    };
  };

  const getDisplayData = () => {
    if (selectedFileId === "all") {
      return {
        metrics: calculateAverageMetrics(),
        posts: allData || allSentimentPosts || [],
        name: "All Datasets",
        isAll: true
      };
    }
    
    return {
      metrics: selectedFile?.evaluationMetrics,
      posts: sentimentPosts || [],
      name: selectedFile?.originalName || "",
      isAll: false
    };
  };

  const { metrics, posts, name, isAll } = getDisplayData();

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
                      <BarChart3Icon className="h-4 w-4 mr-2" />
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

              {selectedFileId && (
                <motion.div 
                  className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg shadow-sm border border-blue-100">
                    <p className="text-xs text-blue-600 font-medium uppercase tracking-wide">Dataset</p>
                    <p className="text-sm font-medium text-slate-800 truncate mt-1">
                      {isAll ? "Combined Analysis (All Datasets)" : name}
                    </p>
                  </div>
                  <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg shadow-sm border border-green-100">
                    <p className="text-xs text-green-600 font-medium uppercase tracking-wide">Records</p>
                    <p className="text-sm font-medium text-slate-800 mt-1">{totalRecords} entries</p>
                  </div>
                  <div className="bg-gradient-to-r from-purple-50 to-fuchsia-50 p-4 rounded-lg shadow-sm border border-purple-100">
                    <p className="text-xs text-purple-600 font-medium uppercase tracking-wide">Analysis Type</p>
                    <p className="text-sm font-medium text-slate-800 mt-1">
                      {isAll 
                        ? "Aggregate sentiment analysis across all files" 
                        : `Individual file analysis (${new Date(selectedFile?.timestamp || "").toLocaleDateString()})`
                      }
                    </p>
                  </div>
                </motion.div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {selectedFileId && (
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid grid-cols-2 w-full max-w-md mx-auto bg-slate-100 p-1 rounded-lg">
            <TabsTrigger value="metrics" className="rounded-md data-[state=active]:bg-white data-[state=active]:text-slate-800 data-[state=active]:shadow-sm">
              Summary Metrics
            </TabsTrigger>
            <TabsTrigger value="confusion" className="rounded-md data-[state=active]:bg-white data-[state=active]:text-slate-800 data-[state=active]:shadow-sm">
              Confusion Matrix
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="metrics" className="space-y-6">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              {/* Metrics Display */}
              <MetricsDisplay 
                data={metrics}
                title={isAll ? "Aggregate Model Performance" : "Model Performance"}
                description={isAll 
                  ? "Combined metrics across all datasets" 
                  : `Metrics for ${name}`
                }
              />
            
              {/* Model Insights */}
              <Card className="bg-white rounded-lg shadow mt-6">
                <CardHeader className="px-6 py-5 border-b border-gray-200">
                  <CardTitle className="text-lg font-medium text-slate-800">Model Insights</CardTitle>
                  <CardDescription className="text-sm text-slate-500">
                    Understanding the sentiment analysis model performance
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                          <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1h4v1a2 2 0 11-4 0zM12 14c.015-.34.208-.646.477-.859a4 4 0 10-4.954 0c.27.213.462.519.476.859h4.002z" />
                        </svg>
                      </div>
                      <div>
                        <h3 className="text-sm font-medium text-slate-800">Accuracy</h3>
                        <p className="mt-1 text-sm text-slate-600">
                          The proportion of total predictions that were correct. Higher is better.
                        </p>
                        {metrics && (
                          <div className="mt-2 flex items-center gap-2">
                            <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-blue-500 rounded-full" 
                                style={{ width: `${metrics.accuracy * 100}%` }} 
                              />
                            </div>
                            <span className="text-xs font-medium text-slate-700 min-w-[40px] text-right">
                              {(metrics.accuracy * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M6.672 1.911a1 1 0 10-1.932.518l.259.966a1 1 0 001.932-.518l-.26-.966zM2.429 4.74a1 1 0 10-.517 1.932l.966.259a1 1 0 00.517-1.932l-.966-.26zm8.814-.569a1 1 0 00-1.415-1.414l-.707.707a1 1 0 101.415 1.415l.707-.708zm-7.071 7.072l.707-.707A1 1 0 003.465 9.12l-.708.707a1 1 0 001.415 1.415zm3.2-5.171a1 1 0 00-1.3 1.3l4 10a1 1 0 001.823.075l1.38-2.759 3.018 3.02a1 1 0 001.414-1.415l-3.019-3.02 2.76-1.379a1 1 0 00-.076-1.822l-10-4z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <div>
                        <h3 className="text-sm font-medium text-slate-800">Precision</h3>
                        <p className="mt-1 text-sm text-slate-600">
                          Of all predicted positive instances, how many were actually positive. Measures false positive rate.
                        </p>
                        {metrics && (
                          <div className="mt-2 flex items-center gap-2">
                            <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-green-500 rounded-full" 
                                style={{ width: `${metrics.precision * 100}%` }} 
                              />
                            </div>
                            <span className="text-xs font-medium text-slate-700 min-w-[40px] text-right">
                              {(metrics.precision * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                          <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                          <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <div>
                        <h3 className="text-sm font-medium text-slate-800">Recall</h3>
                        <p className="mt-1 text-sm text-slate-600">
                          Of all actual positive instances, how many were predicted as positive. Measures false negative rate.
                        </p>
                        {metrics && (
                          <div className="mt-2 flex items-center gap-2">
                            <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-orange-500 rounded-full" 
                                style={{ width: `${metrics.recall * 100}%` }} 
                              />
                            </div>
                            <span className="text-xs font-medium text-slate-700 min-w-[40px] text-right">
                              {(metrics.recall * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <div>
                        <h3 className="text-sm font-medium text-slate-800">F1 Score</h3>
                        <p className="mt-1 text-sm text-slate-600">
                          The harmonic mean of precision and recall. Balances both metrics for an overall performance measure.
                        </p>
                        {metrics && (
                          <div className="mt-2 flex items-center gap-2">
                            <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-purple-500 rounded-full" 
                                style={{ width: `${metrics.f1Score * 100}%` }} 
                              />
                            </div>
                            <span className="text-xs font-medium text-slate-700 min-w-[40px] text-right">
                              {(metrics.f1Score * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>
          
          <TabsContent value="confusion">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              {/* Dynamic Confusion Matrix */}
              <ConfusionMatrix 
                fileId={selectedFileId !== "all" ? parseInt(selectedFileId) : undefined}
                confusionMatrix={selectedFile?.evaluationMetrics?.confusionMatrix}
                title={isAll ? "Aggregate Sentiment Confusion Matrix" : "Sentiment Confusion Matrix"}
                description={isAll 
                  ? "Detailed model prediction accuracy across all datasets" 
                  : `Detailed model prediction accuracy for ${name}`
                }
                allDatasets={selectedFileId === "all"}
              />
            </motion.div>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}