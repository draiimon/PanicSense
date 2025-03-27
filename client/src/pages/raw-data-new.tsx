import { useState, useEffect } from "react";
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
        <div className="relative p-6 z-10">
          <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-700 to-blue-600 bg-clip-text text-transparent">
            Disaster Sentiment Analysis
          </h2>
          <p className="mt-3 text-base text-slate-700">
            View and analyze sentiment data from social media during disasters
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
          className="px-4 py-2 bg-green-600 text-white rounded-lg"
        >
          <Download className="h-4 w-4 mr-2 inline" />
          <span>Download CSV</span>
        </motion.button>
        
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button 
              variant="destructive"
              disabled={isDeleting}
            >
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
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
              <AlertDialogDescription>
                This action will permanently delete all sentiment posts,
                disaster events, and analyzed files from the database.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction onClick={handleDeleteAllData}>
                Yes, Delete All Data
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <FileUploader className="w-auto" />
      </div>

      {/* Filter section */}
      <Card>
        <CardHeader>
          <CardTitle>Data Filters</CardTitle>
          <CardDescription>Filter and analyze specific datasets</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row sm:items-center gap-5">
            <div className="flex-grow">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Filter by Dataset
              </label>
              <Select value={selectedFileId} onValueChange={setSelectedFileId}>
                <SelectTrigger>
                  <SelectValue placeholder="All datasets" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All datasets</SelectItem>
                  {safeFilesArray.map((file) => (
                    <div
                      key={file.id}
                      className="flex justify-between items-center"
                    >
                      <SelectItem value={file.id.toString()}>
                        {file.originalName} ({file.recordCount} records)
                      </SelectItem>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteFile(file.id);
                        }}
                        disabled={deletingFileId === file.id}
                      >
                        {deletingFileId === file.id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <FileX className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  ))}
                </SelectContent>
              </Select>
            </div>
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