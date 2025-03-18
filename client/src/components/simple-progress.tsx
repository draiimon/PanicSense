import { useDisasterContext } from "@/context/disaster-context";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, AlertCircle, FileCheck, Database, ArrowRight, Clock } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";

const EventItem = ({ 
  time, 
  message, 
  status 
}: { 
  time: string; 
  message: string; 
  status: 'pending' | 'active' | 'complete' | 'error' 
}) => (
  <motion.div
    initial={{ opacity: 0, x: -20 }}
    animate={{ opacity: 1, x: 0 }}
    className={`
      flex items-start gap-3 p-3 rounded-lg border
      ${status === 'active' ? 'bg-blue-50 border-blue-200' :
        status === 'complete' ? 'bg-green-50 border-green-200' :
        status === 'error' ? 'bg-red-50 border-red-200' :
        'bg-gray-50 border-gray-200'}
    `}
  >
    <div className="min-w-[40px] flex items-center justify-center">
      {status === 'active' && <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />}
      {status === 'complete' && <FileCheck className="w-5 h-5 text-green-600" />}
      {status === 'error' && <AlertCircle className="w-5 h-5 text-red-600" />}
      {status === 'pending' && <Clock className="w-5 h-5 text-gray-400" />}
    </div>
    <div className="flex-1">
      <div className="text-xs text-gray-500 mb-1">{time}</div>
      <div className="text-sm font-medium">{message}</div>
    </div>
  </motion.div>
);

export function SimpleProgress() {
  const { uploadProgress, isUploading } = useDisasterContext();

  if (!isUploading) return null;

  const { processed, total, stage } = uploadProgress;
  const percentage = total ? Math.round((processed / total) * 100) : 0;

  // Get current time in HH:MM:SS format
  const getCurrentTime = () => {
    const now = new Date();
    return now.toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // Generate event list based on progress
  const getEventList = () => {
    const events = [];
    const time = getCurrentTime();

    if (stage.toLowerCase().includes('loading')) {
      events.push({
        time,
        message: 'Initializing data processing...',
        status: 'complete' as const
      });
      events.push({
        time,
        message: `Loading CSV file for analysis`,
        status: 'active' as const
      });
      events.push({
        time,
        message: 'Preparing sentiment analysis engine',
        status: 'pending' as const
      });
    } else if (stage.toLowerCase().includes('processing')) {
      events.push({
        time,
        message: 'File loaded successfully',
        status: 'complete' as const
      });
      events.push({
        time,
        message: `Processing record ${processed} of ${total}`,
        status: 'active' as const
      });
      events.push({
        time,
        message: 'Preparing final analysis',
        status: 'pending' as const
      });
    } else if (stage.toLowerCase().includes('complete')) {
      events.push({
        time,
        message: 'File loaded successfully',
        status: 'complete' as const
      });
      events.push({
        time,
        message: `Processed ${total} records`,
        status: 'complete' as const
      });
      events.push({
        time,
        message: 'Analysis complete!',
        status: 'complete' as const
      });
    }

    return events;
  };

  return (
    <div className="fixed inset-0 bg-black/20 backdrop-blur-sm flex items-center justify-center z-50">
      <motion.div 
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="bg-white rounded-xl shadow-2xl p-6 w-[500px] max-w-[95vw]"
      >
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-1">
            Processing Dataset
          </h3>
          <div className="text-sm text-gray-600">
            {stage}
          </div>
        </div>

        {/* Timeline Events */}
        <ScrollArea className="h-[300px] mb-6 pr-4">
          <div className="space-y-3">
            {getEventList().map((event, index) => (
              <EventItem
                key={`${event.status}-${index}`}
                time={event.time}
                message={event.message}
                status={event.status}
              />
            ))}
          </div>
        </ScrollArea>

        {/* Progress Bar */}
        <div className="mt-4">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-gray-600">Overall Progress</span>
            <motion.span 
              className="font-semibold text-blue-600"
              key={percentage}
              initial={{ y: 10, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
            >
              {percentage}%
            </motion.span>
          </div>
          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-blue-500"
              initial={{ width: 0 }}
              animate={{ width: `${percentage}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>
      </motion.div>
    </div>
  );
}