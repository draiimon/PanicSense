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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { getSentimentBadgeClasses } from "@/lib/colors";
import { SentimentPost, deleteSentimentPost } from "@/lib/api";
import { format } from "date-fns";
import { Trash2, Search, Filter } from 'lucide-react';
import { useToast } from "@/hooks/use-toast";
import { motion, AnimatePresence } from "framer-motion";
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

const EMOTIONS = ["All", "Panic", "Fear/Anxiety", "Disbelief", "Resilience", "Neutral"];

export function DataTable({ 
  data, 
  title = "Sentiment Analysis Data",
  description = "Raw data from sentiment analysis"
}: DataTableProps) {
  const { toast } = useToast();
  const { refreshData } = useDisasterContext();
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedSentiment, setSelectedSentiment] = useState<string>("All");
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
      item.source?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.location?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.disasterType?.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesSentiment = selectedSentiment === "All" ? true : item.sentiment === selectedSentiment;

    return matchesSearch && matchesSentiment;
  });

  // Pagination
  const totalPages = Math.ceil(filteredData.length / rowsPerPage);
  const startIndex = (currentPage - 1) * rowsPerPage;
  const paginatedData = filteredData.slice(startIndex, startIndex + rowsPerPage);

  return (
    <Card className="bg-white/80 backdrop-blur-sm rounded-lg shadow-lg border border-slate-200/50">
      <CardHeader className="p-6 border-b border-slate-200/50">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
          <div>
            <CardTitle className="text-xl font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              {title}
            </CardTitle>
            <CardDescription className="text-sm text-slate-600">{description}</CardDescription>
          </div>
          <div className="flex flex-col sm:flex-row gap-4">
            
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-500" />
                <Input
                  placeholder="Search in all columns..."
                  value={searchTerm}
                  onChange={(e) => {
                    setSearchTerm(e.target.value);
                    setCurrentPage(1);
                  }}
                  className="pl-9 w-full sm:w-64 bg-white/50 backdrop-blur-sm border-slate-200/50 focus:border-blue-500 transition-all duration-200"
                />
              </div>
            <Select
              value={selectedSentiment}
              onValueChange={(value) => {
                setSelectedSentiment(value);
                setCurrentPage(1);
              }}
            >
              <SelectTrigger className="w-[180px] bg-white/50 backdrop-blur-sm border-slate-200/50">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue placeholder="Filter by emotion" />
              </SelectTrigger>
              <SelectContent>
                {EMOTIONS.map((emotion) => (
                  <SelectItem key={emotion} value={emotion}>
                    {emotion}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="bg-slate-50/50">
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
              <AnimatePresence>
                {paginatedData.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={9} className="text-center py-8 text-slate-500">
                      {searchTerm || selectedSentiment !== "All"
                        ? "No results match your search criteria" 
                        : "No data available"}
                    </TableCell>
                  </TableRow>
                ) : (
                  paginatedData.map((item, index) => (
                    <motion.tr
                      key={item.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.3, delay: index * 0.1 }}
                      className="border-b border-slate-200/50 hover:bg-slate-50/50 transition-colors duration-200"
                    >
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
                        <Badge 
                          variant={getSentimentVariant(item.sentiment) as any}
                          className="animate-fade-in transition-all duration-200"
                        >
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
                              className="h-8 w-8 text-slate-500 hover:text-red-600 transition-colors duration-200"
                              onClick={() => setPostToDelete(item.id)}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </AlertDialogTrigger>
                          <AlertDialogContent className="bg-white/95 backdrop-blur-sm border-slate-200/50">
                            <AlertDialogHeader>
                              <AlertDialogTitle>Delete this post?</AlertDialogTitle>
                              <AlertDialogDescription>
                                This will permanently delete this sentiment post from the database.
                                This action cannot be undone.
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                              <AlertDialogCancel 
                                onClick={() => setPostToDelete(null)}
                                className="hover:bg-slate-100 transition-colors duration-200"
                              >
                                Cancel
                              </AlertDialogCancel>
                              <AlertDialogAction 
                                disabled={isDeleting}
                                onClick={() => postToDelete && handleDeletePost(postToDelete)}
                                className="bg-red-600 hover:bg-red-700 transition-colors duration-200"
                              >
                                {isDeleting ? "Deleting..." : "Delete Post"}
                              </AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>
                      </TableCell>
                    </motion.tr>
                  ))
                )}
              </AnimatePresence>
            </TableBody>
          </Table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between p-4 border-t border-slate-200/50">
            <div className="text-sm text-slate-500">
              Showing {startIndex + 1}-{Math.min(startIndex + rowsPerPage, filteredData.length)} of {filteredData.length} results
            </div>
            <div className="flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="hover:bg-slate-100 transition-colors duration-200"
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
                className="hover:bg-slate-100 transition-colors duration-200"
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