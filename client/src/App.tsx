import { Switch, Route } from "wouter";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import NotFound from "@/pages/not-found";
import Dashboard from "@/pages/dashboard";
import GeographicAnalysis from "@/pages/geographic-analysis";
import Timeline from "@/pages/timeline";
import Comparison from "@/pages/comparison";
import RawData from "@/pages/raw-data";
import Evaluation from "@/pages/evaluation";
import RealTime from "@/pages/real-time";
import About from "@/pages/about";
import { DisasterContextProvider } from "@/context/disaster-context";
import { MainLayout } from "@/components/layout/main-layout";
import { UploadProgressModal } from "@/components/upload-progress-modal";
import { EmergencyResetButton } from "@/components/emergency-reset-button";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/dashboard" component={Dashboard} />
      <Route path="/geographic-analysis" component={GeographicAnalysis} />
      {/* Keep the old route for backward compatibility */}
      <Route path="/emotion-analysis" component={GeographicAnalysis} />
      <Route path="/timeline" component={Timeline} />
      <Route path="/comparison" component={Comparison} />
      <Route path="/raw-data" component={RawData} />
      <Route path="/evaluation" component={Evaluation} />
      <Route path="/real-time" component={RealTime} />
      <Route path="/about" component={About} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <DisasterContextProvider>
        {/* Global upload progress modal to ensure it stays visible across all pages */}
        <UploadProgressModal />
        {/* Emergency reset button for stuck modals - activate with 5 quick Shift key presses */}
        <EmergencyResetButton />
        <MainLayout>
          <Router />
        </MainLayout>
        <Toaster />
      </DisasterContextProvider>
    </QueryClientProvider>
  );
}

export default App;