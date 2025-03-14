import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { format } from 'date-fns';
import { getSentimentColor } from '@/lib/colors';

interface TimelineEvent {
  id: number;
  name: string;
  description: string;
  timestamp: string;
  location: string | null;
  type: string;
  sentimentImpact: string | null;
}

interface KeyEventsProps {
  events: TimelineEvent[];
  title?: string;
  description?: string;
}

export function KeyEvents({ 
  events, 
  title = 'Key Events',
  description = 'Major shifts in sentiment patterns'
}: KeyEventsProps) {
  const getSentimentBadgeText = (sentiment: string | null) => {
    if (!sentiment) return 'Neutral sentiment change';
    
    switch (sentiment) {
      case 'Panic': return 'Panic sentiment spike';
      case 'Fear/Anxiety': return 'Fear/Anxiety sentiment spike';
      case 'Disbelief': return 'Disbelief sentiment increase';
      case 'Resilience': return 'Resilience sentiment increase';
      case 'Neutral':
      default: return 'Neutral sentiment change';
    }
  };

  const getSentimentVariant = (sentiment: string | null) => {
    if (!sentiment) return 'neutral';
    
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
        <CardTitle className="text-lg font-medium text-slate-800">{title}</CardTitle>
        <CardDescription className="text-sm text-slate-500">{description}</CardDescription>
      </CardHeader>
      <CardContent className="p-5 relative">
        {/* Timeline */}
        <div className="absolute top-5 bottom-0 left-5 w-0.5 bg-slate-200" />
        
        {/* Events */}
        <div className="ml-10 space-y-8">
          {events.length === 0 ? (
            <p className="text-slate-500">No key events available</p>
          ) : (
            events.map((event) => {
              const color = getSentimentColor(event.sentimentImpact);
              
              return (
                <div key={event.id} className="relative pl-8">
                  <div 
                    className="absolute -left-14 mt-1.5 w-7 h-7 rounded-full border-4 border-white flex items-center justify-center"
                    style={{ backgroundColor: color }}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 text-white" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <h3 className="text-base font-semibold text-slate-800">{event.name}</h3>
                  <p className="text-sm text-slate-500 mt-1">
                    {format(new Date(event.timestamp), 'MMM d, yyyy - h:mm a')}
                  </p>
                  <p className="text-sm text-slate-600 mt-2">{event.description}</p>
                  <div className="mt-3">
                    <Badge 
                      variant={getSentimentVariant(event.sentimentImpact) as any}
                    >
                      {getSentimentBadgeText(event.sentimentImpact)}
                    </Badge>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </CardContent>
    </Card>
  );
}
