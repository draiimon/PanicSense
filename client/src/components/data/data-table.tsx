import { useState } from "react";
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle, 
  CardDescription 
} from "@/components/ui/card";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { getSentimentBadgeClasses } from "@/lib/colors";
import { SentimentPost, deleteSentimentPost } from "@/lib/api";
import { format } from "date-fns";
import { Trash2 } from 'lucide-react';
import { useToast } from "@/hooks/use-toast";
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
import { useDisasterContext } from "@/context/disaster-context";

interface DataTableProps {
  data: SentimentPost[];
  title?: string;
  description?: string;
}

export function DataTable({ 
  data, 
  title = "Sentiment Analysis Data",
  description = "Raw data from sentiment analysis"
}: DataTableProps) {
  const { toast } = useToast();
  const { refreshData } = useDisasterContext();
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedSentiment, setSelectedSentiment] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [postToDelete, setPostToDelete] = useState<number | null>(null);
  const rowsPerPage = 10;
  
  // Handle delete post
  const handleDeletePost = async (id: number) => {
    try {
      setIsDeleting(true);
      const result = await deleteSentimentPost(id);
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
        description: "Failed to delete post. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsDeleting(false);
      setPostToDelete(null);
    }
  };

  // Filter data based on search term and sentiment filter
  const filteredData = data.filter(item => {
    const matchesSearch = 
      item.text.toLowerCase().includes(searchTerm.toLowerCase()) || 
      item.source?.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesSentiment = selectedSentiment ? item.sentiment === selectedSentiment : true;
    
    return matchesSearch && matchesSentiment;
  });

  // Pagination
  const totalPages = Math.ceil(filteredData.length / rowsPerPage);
  const startIndex = (currentPage - 1) * rowsPerPage;
  const paginatedData = filteredData.slice(startIndex, startIndex + rowsPerPage);

  // Get variant type for sentiment badge
  const getSentimentVariant = (sentiment: string) => {
    switch (sentiment) {
      case 'Panic': return 'panic';
      case 'Fear/Anxiety': return 'fear';
      case 'Disbelief': return 'disbelief';
      case 'Resilience': return 'resilience';
      case 'Neutral': 
      default: return 'neutral';
    }
  };

  return (
    <Card className="bg-white rounded-lg shadow">
      <CardHeader className="p-5 border-b border-gray-200">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
          <div>
            <CardTitle className="text-lg font-medium text-slate-800">{title}</CardTitle>
            <CardDescription className="text-sm text-slate-500">{description}</CardDescription>
          </div>
          <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4">
            <Input
              placeholder="Search by text or source..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1); // Reset to first page on search
              }}
              className="max-w-xs"
            />
            <div className="flex flex-wrap gap-2">
              {["Panic", "Fear/Anxiety", "Disbelief", "Resilience", "Neutral"].map((sentiment) => (
                <Button
                  key={sentiment}
                  variant={selectedSentiment === sentiment ? "default" : "outline"}
                  size="sm"
                  onClick={() => {
                    setSelectedSentiment(selectedSentiment === sentiment ? null : sentiment);
                    setCurrentPage(1); // Reset to first page on filter change
                  }}
                  className={selectedSentiment === sentiment ? getSentimentBadgeClasses(sentiment) : ""}
                >
                  {sentiment}
                </Button>
              ))}
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[30%]">Text</TableHead>
                <TableHead>Timestamp</TableHead>
                <TableHead>Source</TableHead>
                <TableHead>Location</TableHead>
                <TableHead>Disaster</TableHead>
                <TableHead>Sentiment</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Language</TableHead>
                <TableHead className="w-10">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedData.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-8 text-slate-500">
                    {searchTerm || selectedSentiment 
                      ? "No results match your search criteria" 
                      : "No data available"}
                  </TableCell>
                </TableRow>
              ) : (
                paginatedData.map((item) => (
                  <TableRow key={item.id}>
                    <TableCell className="font-medium text-sm text-slate-700">
                      {item.text}
                    </TableCell>
                    <TableCell className="text-sm text-slate-500">
                      {format(new Date(item.timestamp), "yyyy-MM-dd HH:mm")}
                    </TableCell>
                    <TableCell className="text-sm text-slate-500">
                      {item.source || "Unknown"}
                    </TableCell>
                    <TableCell className="text-sm text-slate-500">
                      {item.location || "Unknown"}
                    </TableCell>
                    <TableCell className="text-sm text-slate-500">
                      {item.disasterType || "Not Specified"}
                    </TableCell>
                    <TableCell>
                      <Badge variant={getSentimentVariant(item.sentiment) as any}>
                        {item.sentiment}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm text-slate-500">
                      {(item.confidence * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell className="text-sm text-slate-500">
                      {item.language}
                    </TableCell>
                    <TableCell>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button 
                            variant="ghost" 
                            size="icon"
                            className="h-8 w-8 text-slate-500 hover:text-red-600"
                            onClick={() => setPostToDelete(item.id)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Delete this post?</AlertDialogTitle>
                            <AlertDialogDescription>
                              This will permanently delete this sentiment post from the database.
                              This action cannot be undone.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel onClick={() => setPostToDelete(null)}>Cancel</AlertDialogCancel>
                            <AlertDialogAction 
                              disabled={isDeleting}
                              onClick={() => postToDelete && handleDeletePost(postToDelete)}
                            >
                              {isDeleting ? "Deleting..." : "Delete Post"}
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between p-4 border-t border-gray-200">
            <div className="text-sm text-slate-500">
              Showing {startIndex + 1}-{Math.min(startIndex + rowsPerPage, filteredData.length)} of {filteredData.length} results
            </div>
            <div className="flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
