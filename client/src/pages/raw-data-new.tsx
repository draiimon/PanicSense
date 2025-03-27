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
import { AlertCircle, Download, FileX, Loader2, Trash2 } from "lucide-react";
import { deleteAllData, deleteAnalyzedFile } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

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

  // Data handling with safety checks
  const isLoading = isLoadingSentimentPosts || isLoadingAnalyzedFiles;
  const safePostsArray = Array.isArray(sentimentPosts) ? sentimentPosts : [];
  const safeFilesArray = Array.isArray(analyzedFiles) ? analyzedFiles : [];

  const handleDeleteAllData = async () => {
    try {
      setIsDeleting(true);
      const result = await deleteAllData();
      toast({
        title: "Success",
        description: result.message,
        variant: "default",
      });
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
      if (selectedFileId === fileId.toString()) {
        setSelectedFileId("all");
      }
      toast({
        title: "Success",
        description: result.message,
        variant: "default",
      });
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

  // Loading state
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[70vh]">
        <div className="w-16 h-16 rounded-full bg-gradient-to-r from-purple-100 to-blue-100 flex items-center justify-center mb-4 shadow-md">
          <Loader2 className="h-8 w-8 text-blue-600 animate-spin" />
        </div>
        <h3 className="text-xl font-semibold bg-gradient-to-r from-purple-700 to-blue-600 bg-clip-text text-transparent">Loading data...</h3>
        <p className="text-slate-500 mt-2">Retrieving sentiment analysis information</p>
      </div>
    );
  }

  // Filter posts by file ID
  const filteredPosts = selectedFileId === "all"
    ? safePostsArray
    : safePostsArray.filter(post => post.fileId === parseInt(selectedFileId));

  // Transform for display
  const transformedPosts = filteredPosts.map(post => ({
    ...post,
    language: languageMap[post.language] || post.language,
  }));

  return (
    <div className="space-y-6">
      {/* Header section */}
      <div className="relative overflow-hidden rounded-2xl border border-slate-200/60 shadow-lg bg-gradient-to-r from-purple-100 via-blue-50 to-white">
        <div className="absolute top-0 left-0 w-40 h-40 bg-purple-300/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-0 w-60 h-60 bg-blue-300/20 rounded-full blur-3xl"></div>
        <div className="relative p-6 z-10">
          <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-700 to-blue-600 bg-clip-text text-transparent">
            Disaster Sentiment Analysis
          </h2>
          <p className="mt-3 text-base text-slate-700">
            View and analyze bilingual sentiment data from social media during disasters
          </p>
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex flex-wrap gap-3 mt-4">
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
          <AlertDialogContent className="rounded-xl border-0">
            <AlertDialogHeader className="border-b pb-4">
              <AlertDialogTitle className="text-xl font-bold">Are you absolutely sure?</AlertDialogTitle>
              <AlertDialogDescription className="text-slate-600">
                This action will permanently delete all sentiment posts,
                disaster events, and analyzed files from the database.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter className="pt-4">
              <AlertDialogCancel className="rounded-full hover:bg-slate-100 border-slate-200">
                Cancel
              </AlertDialogCancel>
              <AlertDialogAction
                onClick={handleDeleteAllData}
                className="rounded-full bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-700 hover:to-rose-700"
              >
                Yes, Delete All Data
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <FileUploader className="w-auto" />
      </div>
      
      {/* CSS for animation */}
      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
        .animate-shimmer {
          animation: shimmer 2.5s infinite;
        }
      `}} />

      {/* Filter section */}
      <Card className="bg-white/90 rounded-xl shadow-md border border-slate-200/60 overflow-hidden">
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
          <div className="flex flex-col gap-5">
            {/* Dataset selector */}
            <div className="w-full">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Filter by Dataset
              </label>
              <Select value={selectedFileId} onValueChange={setSelectedFileId}>
                <SelectTrigger className="w-full bg-white/80 backdrop-blur-sm border-slate-200/80 rounded-lg shadow-sm">
                  <SelectValue placeholder="All datasets" />
                </SelectTrigger>
                <SelectContent className="rounded-lg border-slate-200/80 shadow-md">
                  <SelectItem value="all" className="focus:bg-blue-50">All datasets</SelectItem>
                  {safeFilesArray.map((file) => (
                    <SelectItem key={file.id} value={file.id.toString()} className="focus:bg-blue-50">
                      {file.originalName} ({file.recordCount} records)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            {/* Dataset management */}
            {safeFilesArray.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-medium text-slate-700 mb-3">Manage Datasets</h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                  {safeFilesArray.map((file) => (
                    <div 
                      key={`manage-${file.id}`} 
                      className="flex items-center justify-between p-3 bg-white rounded-lg border border-slate-200/60 shadow-sm"
                    >
                      <div className="truncate mr-2">
                        <p className="text-sm font-medium text-slate-800 truncate">{file.originalName}</p>
                        <p className="text-xs text-slate-500">{file.recordCount} records</p>
                      </div>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 w-8 rounded-full text-slate-400 hover:text-red-500 hover:bg-red-50"
                            disabled={deletingFileId === file.id}
                          >
                            {deletingFileId === file.id ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <FileX className="h-4 w-4" />
                            )}
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent className="rounded-xl border-0">
                          <AlertDialogHeader className="border-b pb-4">
                            <AlertDialogTitle className="text-xl font-bold">Delete this file?</AlertDialogTitle>
                            <AlertDialogDescription className="text-slate-600">
                              This will delete "{file.originalName}" and all associated sentiment posts. This action cannot be undone.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter className="pt-4">
                            <AlertDialogCancel className="rounded-full hover:bg-slate-100 border-slate-200">
                              Cancel
                            </AlertDialogCancel>
                            <AlertDialogAction
                              onClick={() => handleDeleteFile(file.id)}
                              className="rounded-full bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-700 hover:to-rose-700"
                            >
                              Delete File
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Data table */}
      {transformedPosts.length > 0 ? (
        <DataTable data={transformedPosts} />
      ) : (
        <div className="flex flex-col items-center justify-center p-8 bg-white rounded-xl shadow">
          <AlertCircle className="w-12 h-12 text-amber-500 mb-4" />
          <h3 className="text-xl font-semibold text-slate-800">No Data Available</h3>
          <p className="text-slate-500 mt-2 text-center">
            There are no sentiment posts to display. Try uploading a CSV file to analyze sentiment data.
          </p>
        </div>
      )}
    </div>
  );
}