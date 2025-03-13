import { Switch, Route } from "wouter";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import NotFound from "@/pages/not-found";
import Dashboard from "@/pages/dashboard";
import EmotionAnalysis from "@/pages/emotion-analysis";
import Timeline from "@/pages/timeline";
import Comparison from "@/pages/comparison";
import RawData from "@/pages/raw-data";
import Evaluation from "@/pages/evaluation";
import About from "@/pages/about";
import { DisasterContextProvider } from "@/context/disaster-context";
import { MainLayout } from "@/components/layout/main-layout";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/dashboard" component={Dashboard} />
      <Route path="/emotion-analysis" component={EmotionAnalysis} />
      <Route path="/timeline" component={Timeline} />
      <Route path="/comparison" component={Comparison} />
      <Route path="/raw-data" component={RawData} />
      <Route path="/evaluation" component={Evaluation} />
      <Route path="/about" component={About} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <DisasterContextProvider>
        <MainLayout>
          <Router />
        </MainLayout>
        <Toaster />
      </DisasterContextProvider>
    </QueryClientProvider>
  );
}

export default App;