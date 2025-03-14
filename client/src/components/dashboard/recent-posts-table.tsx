import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Link } from "wouter";
import { format } from "date-fns";

interface Post {
  id: number;
  text: string;
  timestamp: string;
  source: string;
  sentiment: string;
  confidence: number;
}

interface RecentPostsTableProps {
  posts?: Post[];
  title?: string;
  description?: string;
  limit?: number;
  showViewAllLink?: boolean;
}

export function RecentPostsTable({ 
  posts = [], 
  title = 'Recent Analyzed Posts',
  description = 'Latest social media sentiment',
  limit = 5,
  showViewAllLink = true
}: RecentPostsTableProps) {
  const displayedPosts = posts?.slice(0, limit) || [];

  const getSentimentVariant = (sentiment: string) => {
    switch (sentiment?.toLowerCase()) {
      case 'panic': return 'destructive';
      case 'fear/anxiety': return 'warning';
      case 'disbelief': return 'secondary';
      case 'resilience': return 'success';
      default: return 'default';
    }
  };

  return (
    <Card className="bg-white rounded-lg shadow">
      <CardHeader className="p-5 border-b border-gray-200 flex justify-between items-center flex-wrap sm:flex-nowrap">
        <div>
          <CardTitle className="text-lg font-medium text-slate-800">{title}</CardTitle>
          <CardDescription className="text-sm text-slate-500">{description}</CardDescription>
        </div>
        {showViewAllLink && (
          <Link href="/raw-data" className="text-sm font-medium text-blue-600 hover:text-blue-800">
            View all
          </Link>
        )}
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200">
            <thead className="bg-slate-50">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Post</th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Timestamp</th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Source</th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Sentiment</th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Confidence</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-slate-200">
              {displayedPosts.map((post) => (
                <tr key={post.id}>
                  <td className="px-6 py-4 text-sm text-slate-900 max-w-xs truncate">{post.text}</td>
                  <td className="px-6 py-4 text-sm text-slate-500">{format(new Date(post.timestamp), 'MMM d, yyyy')}</td>
                  <td className="px-6 py-4 text-sm text-slate-500">{post.source}</td>
                  <td className="px-6 py-4">
                    <Badge variant={getSentimentVariant(post.sentiment)}>{post.sentiment}</Badge>
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-500">{Math.round(post.confidence * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}