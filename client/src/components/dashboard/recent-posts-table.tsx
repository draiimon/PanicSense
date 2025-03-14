import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { SentimentPost } from "@/types";

interface RecentPostsTableProps {
  posts?: SentimentPost[];
  limit?: number;
  title?: string;
  description?: string;
}

export function RecentPostsTable({
  posts = [],
  title = "Recent Posts",
  description = "Latest analyzed social media posts",
  limit = 5
}: RecentPostsTableProps) {
  const displayedPosts = posts?.slice(0, limit) || [];

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        {displayedPosts.length === 0 ? (
          <div className="text-center py-8 text-slate-500">
            No posts available
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Text</TableHead>
                <TableHead>Sentiment</TableHead>
                <TableHead>Date</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {displayedPosts.map((post) => (
                <TableRow key={post.id}>
                  <TableCell className="max-w-[300px] truncate">{post.text}</TableCell>
                  <TableCell>
                    <Badge variant="secondary">{post.sentiment}</Badge>
                  </TableCell>
                  <TableCell>{new Date(post.timestamp).toLocaleDateString()}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}