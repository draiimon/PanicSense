import { RealtimeMonitor } from "@/components/realtime/realtime-monitor";

export default function RealTime() {
  return (
    <div className="space-y-6">
      {/* Real-Time Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Real-Time Mode</h1>
        <p className="mt-1 text-sm text-slate-500">
          Analyze disaster-related text in real-time with detailed sentiment explanations
        </p>
      </div>

      {/* Realtime Monitor Component */}
      <RealtimeMonitor />

      {/* Instructions Card */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-medium text-slate-800 mb-3">
          How to Use Real-Time Analysis
        </h2>
        <div className="space-y-4">
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
              <span className="text-blue-600 font-medium">1</span>
            </div>
            <p className="text-sm text-slate-700">
              Enter disaster-related text in the input field on the left.
            </p>
          </div>
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
              <span className="text-blue-600 font-medium">2</span>
            </div>
            <p className="text-sm text-slate-700">
              Click the "Analyze Sentiment" button to process the text using our AI model.
            </p>
          </div>
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
              <span className="text-blue-600 font-medium">3</span>
            </div>
            <p className="text-sm text-slate-700">
              View the results in the right panel, showing the detected sentiment and confidence level.
            </p>
          </div>
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
              <span className="text-blue-600 font-medium">4</span>
            </div>
            <p className="text-sm text-slate-700">
              Continue adding more text samples to build a real-time analysis history.
            </p>
          </div>
        </div>

        <div className="mt-6 p-4 bg-amber-50 border border-amber-100 rounded-lg">
          <div className="flex items-start">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-amber-500 mt-0.5 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            <div>
              <h3 className="text-sm font-medium text-amber-800">Tips for better analysis</h3>
              <p className="mt-1 text-sm text-amber-700">
                For more accurate results, provide detailed context in your text. Mention specific disaster types, locations, and emotional reactions. The model works best with text between 20-200 words.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
