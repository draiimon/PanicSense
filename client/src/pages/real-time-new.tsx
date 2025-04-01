import { RealtimeMonitor } from "@/components/realtime/realtime-monitor";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Lightbulb, Zap, MessageSquareText, ClipboardCheck } from "lucide-react";

export default function RealTime() {
  // Animation variants for staggered animation
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { type: "spring", stiffness: 100 }
    }
  };

  return (
    <div className="relative min-h-screen">
      {/* Enhanced background - EXACTLY LIKE DASHBOARD */}
      <div className="fixed inset-0 -z-10 bg-gradient-to-b from-violet-50 to-pink-50 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-500/15 via-teal-500/10 to-rose-500/15 animate-gradient"
          style={{ backgroundSize: "200% 200%" }} />
          
        <div className="absolute inset-0 opacity-15 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiM1MDUwRjAiIGZpbGwtb3BhY2l0eT0iMC41Ij48cGF0aCBkPSJNMzYgMzR2Nmg2di02aC02em02IDZ2Nmg2di02aC02em0tMTIgMGg2djZoLTZ2LTZ6bTEyIDBoNnY2aC02di02eiIvPjwvZz48L2c+PC9zdmc+')]" />
        
        <div className="absolute inset-0 opacity-10 bg-[radial-gradient(circle_at_center,rgba(120,80,255,0.8)_0%,transparent_70%)]" />
        
        <div className="absolute h-72 w-72 rounded-full bg-purple-500/25 filter blur-3xl animate-float-1 will-change-transform"
          style={{ top: "15%", left: "8%" }} />
          
        <div className="absolute h-64 w-64 rounded-full bg-teal-500/20 filter blur-3xl animate-float-2 will-change-transform"
          style={{ bottom: "15%", right: "15%" }} />
          
        <div className="absolute h-52 w-52 rounded-full bg-purple-500/25 filter blur-3xl animate-float-3 will-change-transform"
          style={{ top: "45%", right: "20%" }} />
        
        <div className="absolute h-48 w-48 rounded-full bg-pink-500/20 filter blur-3xl animate-float-4 will-change-transform"
          style={{ top: "65%", left: "25%" }} />
          
        <div className="absolute h-40 w-40 rounded-full bg-yellow-400/15 filter blur-3xl animate-float-5 will-change-transform"
          style={{ top: "30%", left: "40%" }} />
          
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent" />
      </div>
      
      <div className="relative pb-10">
        <motion.div 
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          className="relative space-y-8 mx-auto max-w-7xl pt-10 px-4"
        >
          {/* Real-Time Header */}
          <motion.div 
            variants={itemVariants}
            className="relative overflow-hidden rounded-2xl border-0 shadow-lg bg-gradient-to-r from-violet-100/90 to-blue-100/90 backdrop-blur-sm p-6"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 via-blue-500/10 to-purple-500/10 animate-gradient" />
            <div className="absolute top-0 left-0 w-40 h-40 bg-violet-400/20 rounded-full blur-3xl" />
            <div className="absolute bottom-0 right-0 w-60 h-60 bg-blue-400/20 rounded-full blur-3xl" />
            
            <div className="relative z-10 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-violet-700 to-blue-600 bg-clip-text text-transparent">
                  Real-Time Analysis
                </h1>
                <p className="mt-3 text-base text-slate-700">
                  Analyze disaster-related text in real-time with detailed emotion analysis and insights
                </p>
              </div>
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-full bg-gradient-to-br from-violet-500 to-blue-600 flex items-center justify-center shadow-md">
                  <Zap className="h-5 w-5 text-white" />
                </div>
                <div className="text-sm font-medium text-slate-600">
                  Instant AI-powered sentiment detection
                </div>
              </div>
            </div>
          </motion.div>

          {/* Realtime Monitor Component */}
          <motion.div variants={itemVariants}>
            <RealtimeMonitor />
          </motion.div>

          {/* Instructions Card */}
          <motion.div variants={itemVariants}>
            <Card className="overflow-hidden shadow-lg border-0 bg-white/90 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-violet-50 to-indigo-50 border-b border-gray-200/40 pb-4">
                <div className="flex justify-between items-center">
                  <div>
                    <CardTitle className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                      <ClipboardCheck className="h-5 w-5 text-violet-600" />
                      How to Use Real-Time Analysis
                    </CardTitle>
                    <CardDescription className="text-slate-500">
                      Follow these steps to get the most out of the real-time sentiment analyzer
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-6">
                <div className="space-y-5">
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-violet-100 flex items-center justify-center shadow-sm">
                      <span className="text-violet-600 font-medium">1</span>
                    </div>
                    <div>
                      <h3 className="text-base font-medium text-slate-800">Enter Text</h3>
                      <p className="mt-1 text-sm text-slate-600">
                        Enter disaster-related text in the input field on the left. You can type in English or Filipino.
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-violet-100 flex items-center justify-center shadow-sm">
                      <span className="text-violet-600 font-medium">2</span>
                    </div>
                    <div>
                      <h3 className="text-base font-medium text-slate-800">Process Analysis</h3>
                      <p className="mt-1 text-sm text-slate-600">
                        Click the "Analyze Sentiment" button to process the text using our advanced AI model, or enable auto-analyze mode.
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-violet-100 flex items-center justify-center shadow-sm">
                      <span className="text-violet-600 font-medium">3</span>
                    </div>
                    <div>
                      <h3 className="text-base font-medium text-slate-800">View Results</h3>
                      <p className="mt-1 text-sm text-slate-600">
                        View the results in the right panel, showing the detected sentiment, confidence level, and additional insights.
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-violet-100 flex items-center justify-center shadow-sm">
                      <span className="text-violet-600 font-medium">4</span>
                    </div>
                    <div>
                      <h3 className="text-base font-medium text-slate-800">Build Analysis History</h3>
                      <p className="mt-1 text-sm text-slate-600">
                        Continue adding more text samples to build a comprehensive real-time analysis history that you can export or review.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="mt-8 p-5 bg-gradient-to-r from-amber-50 to-yellow-50 border border-amber-200/60 rounded-lg shadow-sm">
                  <div className="flex items-start space-x-3">
                    <Lightbulb className="h-6 w-6 text-amber-500 flex-shrink-0" />
                    <div>
                      <h3 className="text-sm font-medium text-amber-800">Tips for Better Analysis</h3>
                      <p className="mt-2 text-sm text-amber-700">
                        For more accurate results, provide detailed context in your text. Mention specific disaster types (earthquake, flood, typhoon, etc.), locations, and emotional reactions. The model works best with text between 20-200 words.
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                        <div className="bg-white/60 backdrop-blur-sm p-3 rounded border border-amber-200/60 text-sm text-amber-800">
                          <span className="font-medium">✓ Good Example:</span> "May nagsasabing nagkakaroon ng lindol sa Maynila. Natatakot ako at hindi ko alam kung safe na lumabas ng bahay."
                        </div>
                        <div className="bg-white/60 backdrop-blur-sm p-3 rounded border border-amber-200/60 text-sm text-amber-800">
                          <span className="font-medium">✓ Good Example:</span> "The flood in Bacolod City is getting worse. People are trapped on rooftops waiting for rescue. I'm worried about my family there."
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}