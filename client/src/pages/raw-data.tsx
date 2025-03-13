import { useState } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { DataTable } from "@/components/data/data-table";
import { FileUploader } from "@/components/file-uploader";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { AlertCircle } from "lucide-react";

// Language mapping
const languageMap: Record<string, string> = {
  'en': 'English',
  'tl': 'Tagalog'
};

export default function RawData() {
  const { 
    sentimentPosts, 
    analyzedFiles, 
    isLoadingSentimentPosts,
    isLoadingAnalyzedFiles
  } = useDisasterContext();
  const [selectedFileId, setSelectedFileId] = useState<string>("all");

  // Filter posts by file ID if selected
  const filteredPosts = selectedFileId === "all" 
    ? sentimentPosts 
    : sentimentPosts.filter(post => post.fileId === parseInt(selectedFileId));

  const isLoading = isLoadingSentimentPosts || isLoadingAnalyzedFiles;

  // Transform posts to display full language names
  const transformedPosts = filteredPosts.map(post => ({
    ...post,
    language: languageMap[post.language] || post.language
  }));

  return (
    <div className="space-y-6">
      {/* Study Significance Section */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-none">
        <CardHeader>
          <div className="flex items-start space-x-2">
            <AlertCircle className="h-5 w-5 text-blue-600 mt-1" />
            <div>
              <CardTitle className="text-xl text-blue-900">Disaster Sentiment Analysis</CardTitle>
              <CardDescription className="text-sm text-blue-800 mt-2 leading-relaxed">
                This analysis system examines bilingual (English and Filipino) social media data during natural disasters,
                providing real-time insights into public emotional responses. The data helps disaster management teams
                understand and respond to public sentiment patterns, enabling better crisis communication and response strategies.
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="bg-white/50 p-4 rounded-lg">
              <h3 className="font-medium text-blue-900">Primary Focus</h3>
              <p className="mt-1 text-blue-800">
                Analyzing social media responses to map emotional patterns during disasters,
                helping authorities better understand and respond to public needs.
              </p>
            </div>
            <div className="bg-white/50 p-4 rounded-lg">
              <h3 className="font-medium text-blue-900">Key Beneficiaries</h3>
              <p className="mt-1 text-blue-800">
                NDRRMC, Public Information Agencies, Local Government Units, and Community Leaders
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Raw Data Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Raw Data Analysis</h1>
          <p className="mt-1 text-sm text-slate-500">
            View and analyze bilingual sentiment data from social media during disasters
          </p>
        </div>
        <FileUploader 
          className="mt-4 sm:mt-0"
          onSuccess={() => {
            // The disaster context will handle refetching data
          }}
        />
      </div>

      {/* Filter Controls */}
      <Card className="bg-white shadow">
        <CardHeader className="border-b border-gray-100">
          <CardTitle className="text-lg">Data Filters</CardTitle>
          <CardDescription>Filter and analyze specific datasets</CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          <div className="flex flex-col sm:flex-row sm:items-center space-y-4 sm:space-y-0 sm:space-x-4">
            <div className="flex-grow">
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Filter by Dataset
              </label>
              <Select
                value={selectedFileId}
                onValueChange={setSelectedFileId}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="All datasets" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All datasets</SelectItem>
                  {analyzedFiles.map((file) => (
                    <SelectItem key={file.id} value={file.id.toString()}>
                      {file.originalName} ({file.recordCount} records)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="text-sm text-slate-500 flex items-center">
              {isLoading ? (
                <div className="flex items-center">
                  <svg className="animate-spin h-4 w-4 mr-2 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Loading data...
                </div>
              ) : (
                <span>
                  Showing {transformedPosts.length} of {sentimentPosts.length} total records
                </span>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Data Table */}
      <DataTable 
        data={transformedPosts}
        title={selectedFileId === "all" ? "Complete Sentiment Dataset" : `Data from ${analyzedFiles.find(f => f.id.toString() === selectedFileId)?.originalName || "Selected File"}`}
        description="Bilingual sentiment analysis results with filtering capabilities"
      />
    </div>
  );
}