import { useState } from "react";
import { useDisasterContext } from "@/context/disaster-context";
import { DataTable } from "@/components/data/data-table";
import { FileUploader } from "@/components/file-uploader";
import { motion } from "framer-motion";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { AlertCircle } from "lucide-react";
import { deleteAllData, deleteAnalyzedFile } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Trash2, FileX, Download } from "lucide-react";

// Language mapping
const languageMap: Record<string, string> = {
  en: "English",
  tl: "Filipino",
};

export default function RawData() {
  const { toast } = useToast();
  const {
    sentimentPosts,
    analyzedFiles,
    isLoadingSentimentPosts,
    isLoadingAnalyzedFiles,
    refreshData,
  } = useDisasterContext();
  const [selectedFileId, setSelectedFileId] = useState<string>("all");
  const [isDeleting, setIsDeleting] = useState(false);
  const [deletingFileId, setDeletingFileId] = useState<number | null>(null);

  const handleDeleteAllData = async () => {
    try {
      setIsDeleting(true);
      const result = await deleteAllData();
      toast({
        title: "Success",
        description: result.message,
        variant: "default",
      });
      // Refresh the data
      refreshData();
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to delete data. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDeleteFile = async (fileId: number) => {
    try {
      setDeletingFileId(fileId);
      const result = await deleteAnalyzedFile(fileId);

      // If current selected file was deleted, reset to 'all'
      if (selectedFileId === fileId.toString()) {
        setSelectedFileId("all");
      }

      toast({
        title: "Success",
        description: result.message,
        variant: "default",
      });
      // Refresh the data
      refreshData();
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to delete file. Please try again.",
        variant: "destructive",
      });
    } finally {
      setDeletingFileId(null);
    }
  };

  // Filter posts by file ID if selected
  const filteredPosts =
    selectedFileId === "all"
      ? sentimentPosts
      : sentimentPosts.filter(
          (post) => post.fileId === parseInt(selectedFileId),
        );

  const isLoading = isLoadingSentimentPosts || isLoadingAnalyzedFiles;

  // Transform posts to display full language names
  const transformedPosts = filteredPosts.map((post) => ({
    ...post,
    language: languageMap[post.language] || post.language,
  }));

  return (
    <div className="space-y-6">
      {/* Study Significance Section - Enhanced Design */}
      <div className="relative overflow-hidden rounded-2xl border border-slate-200/60 shadow-lg bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-purple-100 via-blue-50 to-white">
        {/* Animated background elements */}
        <div className="absolute top-0 left-0 w-40 h-40 bg-purple-300/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-0 w-60 h-60 bg-blue-300/20 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 right-1/4 w-20 h-20 bg-indigo-300/30 rounded-full blur-2xl"></div>
        
        <div className="relative p-6 z-10">
          <div>
            <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-700 to-blue-600 bg-clip-text text-transparent">
              Disaster Sentiment Analysis
            </h2>
            <p className="mt-3 text-base text-slate-700 leading-relaxed">
              This advanced analysis system examines bilingual (English and Filipino)
              social media data during natural disasters, providing real-time
              insights into public emotional responses. The data helps
              disaster management teams understand and respond to public
              sentiment patterns, enabling better crisis communication and
              response strategies.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div className="bg-white/80 backdrop-blur-sm p-5 rounded-xl shadow-sm border border-slate-200/60">
              <div className="flex items-center gap-3 mb-3">
                <div className="h-8 w-8 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 flex items-center justify-center shadow-sm">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white">
                    <path d="M18 6 7 12l11 6z"/>
                    <path d="m7 12 11 6"/>
                  </svg>
                </div>
                <h3 className="font-semibold text-lg text-slate-800">Primary Focus</h3>
              </div>
              <p className="text-slate-600 leading-relaxed">
                Analyzing social media responses to map emotional patterns
                during disasters, helping authorities better understand and
                respond to public needs in real-time through advanced sentiment analysis.
              </p>
            </div>
            
            <div className="bg-white/80 backdrop-blur-sm p-5 rounded-xl shadow-sm border border-slate-200/60">
              <div className="flex items-center gap-3 mb-3">
                <div className="h-8 w-8 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 flex items-center justify-center shadow-sm">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white">
                    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                    <circle cx="9" cy="7" r="4"/>
                    <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
                    <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
                  </svg>
                </div>
                <h3 className="font-semibold text-lg text-slate-800">Key Beneficiaries</h3>
              </div>
              <ul className="space-y-2 text-slate-600">
                <li className="flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-blue-500"></span>
                  <span>NDRRMC & Disaster Response Teams</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-purple-500"></span>
                  <span>Public Information & Media Agencies</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-indigo-500"></span>
                  <span>Local Government Units & Community Leaders</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Raw Data Header with Enhanced Design */}
      <div className="relative mb-8 overflow-hidden p-6 bg-gradient-to-r from-purple-50 via-indigo-50 to-blue-50 rounded-2xl shadow-md border border-slate-200/60">
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>

        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6 relative z-10">
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-700 to-blue-600 bg-clip-text text-transparent">
              Raw Data Analysis
            </h1>
            <p className="mt-2 text-sm text-slate-600">
              View and analyze bilingual sentiment data from social media during disasters
            </p>
          </div>
          
          <div className="flex flex-wrap gap-3 mt-2 md:mt-0">
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={async () => {
                try {
                  const response = await fetch("/api/export-csv");
                  if (!response.ok) throw new Error("Failed to download data");

                  const blob = await response.blob();
                  const url = window.URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = "disaster-sentiments.csv";
                  document.body.appendChild(a);
                  a.click();
                  window.URL.revokeObjectURL(url);
                  document.body.removeChild(a);

                  toast({
                    title: "Success",
                    description: "Data exported successfully",
                    variant: "default",
                  });
                } catch (error) {
                  toast({
                    title: "Error",
                    description: "Failed to export data",
                    variant: "destructive",
                  });
                }
              }}
              className="relative inline-flex items-center justify-center px-5 py-2.5 h-10 rounded-full bg-gradient-to-r from-emerald-600 to-green-600 text-white shadow-md hover:shadow-lg overflow-hidden"
            >
              {/* Animated shimmer effect */}
              <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-white/0 via-white/25 to-white/0 animate-shimmer -translate-x-full"></div>
              {/* Content */}
              <div className="flex items-center justify-center">
                <Download className="h-4 w-4 mr-2" />
                <span>Download CSV</span>
              </div>
            </motion.button>
            
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <motion.button
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                  disabled={isDeleting}
                  className="relative inline-flex items-center justify-center px-5 py-2.5 h-10 rounded-full bg-gradient-to-r from-red-600 to-rose-600 text-white shadow-md hover:shadow-lg overflow-hidden"
                >
                  {/* Animated shimmer effect */}
                  <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-white/0 via-white/25 to-white/0 animate-shimmer -translate-x-full"></div>
                  {/* Content */}
                  <div className="flex items-center justify-center">
                    {isDeleting ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        <span>Deleting...</span>
                      </>
                    ) : (
                      <>
                        <Trash2 className="h-4 w-4 mr-2" />
                        <span>Delete All Data</span>
                      </>
                    )}
                  </div>
                </motion.button>
              </AlertDialogTrigger>
              <AlertDialogContent className="bg-white/95 backdrop-blur-sm border-slate-200/60 rounded-xl shadow-xl">
                <AlertDialogHeader>
                  <AlertDialogTitle className="text-xl">Are you absolutely sure?</AlertDialogTitle>
                  <AlertDialogDescription className="text-slate-600">
                    This action will permanently delete all sentiment posts,
                    disaster events, and analyzed files from the database. This
                    action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel className="rounded-full hover:bg-slate-100">
                    Cancel
                  </AlertDialogCancel>
                  <AlertDialogAction
                    onClick={handleDeleteAllData}
                    className="rounded-full bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-700 hover:to-rose-700 shadow-md"
                  >
                    Yes, Delete All Data
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>

            <FileUploader className="w-auto" />
          </div>
        </div>
      </div>

      {/* CSS for Grid Pattern */}
      <style dangerouslySetInnerHTML={{ __html: `
        .bg-grid-pattern {
          background-image: 
            linear-gradient(to right, rgba(0,0,0,0.05) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(0,0,0,0.05) 1px, transparent 1px);
          background-size: 20px 20px;
        }
        
        @keyframes shimmer {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(200%);
          }
        }
        .animate-shimmer {
          animation: shimmer 2.5s infinite;
        }
      `}} />

      {/* Filter Controls - Enhanced Design */}
      <Card className="bg-white/90 rounded-xl shadow-lg border border-slate-200/60 overflow-hidden">
        <CardHeader className="p-5 bg-gradient-to-r from-slate-50 to-indigo-50/50">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-full bg-gradient-to-br from-indigo-500 to-blue-600 flex items-center justify-center shadow-md">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>
            </div>
            <div>
              <CardTitle className="text-lg bg-gradient-to-r from-indigo-700 to-blue-600 bg-clip-text text-transparent">Data Filters</CardTitle>
              <CardDescription className="text-slate-600 mt-1">
                Filter and analyze specific datasets
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-5">
          <div className="flex flex-col sm:flex-row sm:items-center gap-5">
            <div className="flex-grow">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Filter by Dataset
              </label>
              <Select value={selectedFileId} onValueChange={setSelectedFileId}>
                <SelectTrigger className="w-full bg-white/80 backdrop-blur-sm border-slate-200/80 rounded-lg shadow-sm">
                  <SelectValue placeholder="All datasets" />
                </SelectTrigger>
                <SelectContent className="min-w-[350px] rounded-lg border-slate-200/80 shadow-md">
                  <SelectItem value="all" className="focus:bg-blue-50">All datasets</SelectItem>
                  {analyzedFiles.map((file) => (
                    <div
                      key={file.id}
                      className="flex justify-between items-center"
                    >
                      <SelectItem
                        value={file.id.toString()}
                        className="flex-grow focus:bg-blue-50"
                      >
                        {file.originalName} ({file.recordCount} records)
                      </SelectItem>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 w-8 px-0 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-full transition-colors"
                            onClick={(e) => e.stopPropagation()}
                            disabled={deletingFileId === file.id}
                          >
                            {deletingFileId === file.id ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <FileX className="h-4 w-4" />
                            )}
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent className="bg-white/95 backdrop-blur-sm border-slate-200/60 rounded-xl shadow-xl">
                          <AlertDialogHeader>
                            <AlertDialogTitle className="text-xl">
                              Delete this file?
                            </AlertDialogTitle>
                            <AlertDialogDescription className="text-slate-600">
                              This will delete the file "{file.originalName}"
                              and all sentiment posts associated with it. This
                              action cannot be undone.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel className="rounded-full hover:bg-slate-100">
                              Cancel
                            </AlertDialogCancel>
                            <AlertDialogAction
                              onClick={() => handleDeleteFile(file.id)}
                              className="rounded-full bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-700 hover:to-rose-700 shadow-md"
                            >
                              Delete
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex h-10 items-center justify-center px-4 bg-white rounded-lg border border-slate-200/60 shadow-sm">
              {isLoading ? (
                <div className="flex items-center text-blue-600">
                  <svg
                    className="animate-spin h-4 w-4 mr-2"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  <span className="text-sm font-medium">Loading data...</span>
                </div>
              ) : (
                <div className="flex items-center">
                  <div className="flex items-center justify-center h-7 w-7 rounded-full bg-indigo-100 mr-2">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-indigo-600">
                      <path d="M20 6 9 17l-5-5"/>
                    </svg>
                  </div>
                  <span className="text-sm font-medium text-slate-700">
                    {transformedPosts.length === sentimentPosts.length ? (
                      `${transformedPosts.length} total records`
                    ) : (
                      `${transformedPosts.length} filtered records`
                    )}
                  </span>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Data Table - Main Focus */}
      <div className="w-full p-1">
        <div className="relative w-full mx-auto">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-200/10 via-blue-200/5 to-white/0 rounded-3xl blur-xl"></div>
          <div className="relative">
            <DataTable
              data={transformedPosts}
              title={
                selectedFileId === "all"
                  ? "Complete Sentiment Dataset"
                  : `Data from ${analyzedFiles.find((f) => f.id.toString() === selectedFileId)?.originalName || "Selected File"}`
              }
              description="Bilingual sentiment analysis results with filtering capabilities"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
