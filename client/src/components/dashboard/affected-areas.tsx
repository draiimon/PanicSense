import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { getSentimentColor } from '@/lib/colors';

interface AffectedArea {
  name: string;
  percentage: number;
  sentiment: string;
}

interface AffectedAreasProps {
  areas: AffectedArea[];
  title?: string;
  description?: string;
}

export function AffectedAreas({ 
  areas, 
  title = 'Top Affected Areas',
  description = 'By sentiment intensity'
}: AffectedAreasProps) {
  return (
    <Card className="bg-white rounded-lg shadow">
      <CardHeader className="p-5 border-b border-gray-200">
        <CardTitle className="text-lg font-medium text-slate-800">{title}</CardTitle>
        <CardDescription className="text-sm text-slate-500">{description}</CardDescription>
      </CardHeader>
      <CardContent className="p-5 space-y-4">
        {areas.length === 0 ? (
          <p className="text-center text-slate-500 py-4">No affected areas data available</p>
        ) : (
          areas.map((area, index) => {
            const color = getSentimentColor(area.sentiment);
            
            return (
              <div key={index} className="flex items-center">
                <div 
                  className="w-2 h-2 rounded-full mr-2"
                  style={{ backgroundColor: color }}
                />
                <div className="flex-1">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-slate-700">{area.name}</span>
                    <span className="text-xs font-medium text-slate-500">{area.percentage}%</span>
                  </div>
                  <div className="mt-1 w-full bg-slate-200 rounded-full h-2">
                    <div 
                      className="h-2 rounded-full" 
                      style={{ 
                        width: `${area.percentage}%`,
                        backgroundColor: color
                      }}
                    />
                  </div>
                </div>
              </div>
            );
          })
        )}
      </CardContent>
    </Card>
  );
}
