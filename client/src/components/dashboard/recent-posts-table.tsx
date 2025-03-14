
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Link } from "wouter";

interface RecentPostsTableProps {
  posts?: Array<{
    id: string;
    content: string;
    sentiment: string;
    timestamp: string;
  }>;
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
    const variants: Record<string, string> = {
      'Panic': 'destructive',
      'Fear/Anxiety': 'warning',
      'Disbelief': 'secondary',
      'Resilience': 'success',
      'Neutral': 'outline'
    };
    return variants[sentiment] || 'default';
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Content</TableHead>
              <TableHead>Sentiment</TableHead>
              <TableHead>Time</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {displayedPosts.map((post) => (
              <TableRow key={post.id}>
                <TableCell className="font-medium">{post.content}</TableCell>
                <TableCell>
                  <Badge variant={getSentimentVariant(post.sentiment)}>
                    {post.sentiment}
                  </Badge>
                </TableCell>
                <TableCell>{new Date(post.timestamp).toLocaleString()}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {showViewAllLink && posts.length > limit && (
          <div className="mt-4 text-right">
            <Button variant="link" asChild>
              <Link href="/posts">View all posts</Link>
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
