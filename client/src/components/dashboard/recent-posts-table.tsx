import { Badge } from "@/components/ui/badge";
import { format } from "date-fns";
import { Link } from "wouter";

interface RecentPostsTableProps {
  data: any[];
  title?: string;
  description?: string;
  limit?: number;
  showViewAllLink?: boolean;
}

export function RecentPostsTable({ 
  data = [],
  title = 'Recent Analyzed Posts',
  description = 'Latest social media sentiment',
  limit = 5,
  showViewAllLink = true
}: RecentPostsTableProps) {
  const displayedPosts = data.slice(0, limit);

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
    <div className="overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200">
          <thead className="bg-slate-50">
            <tr>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Content</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Time</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Source</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Sentiment</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Confidence</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-slate-200">
            {displayedPosts.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-6 py-4 text-center text-sm text-slate-500">
                  No posts available
                </td>
              </tr>
            ) : (
              displayedPosts.map((post) => (
                <tr key={post.id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <p className="text-sm text-slate-700 line-clamp-1">{post.text}</p>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                    {format(new Date(post.timestamp), 'yyyy-MM-dd HH:mm')}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                    {post.source || 'Unknown'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Badge variant={getSentimentVariant(post.sentiment)}>
                      {post.sentiment}
                    </Badge>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                    {(post.confidence * 100).toFixed(1)}%
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      {showViewAllLink && (
        <div className="mt-4 text-right">
          <Link href="/raw-data" className="text-sm text-blue-600 hover:text-blue-800">
            View all posts →
          </Link>
        </div>
      )}
    </div>
  );
}