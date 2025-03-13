import { Switch, Route } from "wouter";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import { AuthProvider, useAuth } from "@/context/auth-context";
import NotFound from "@/pages/not-found";
import Dashboard from "@/pages/dashboard";
import EmotionAnalysis from "@/pages/emotion-analysis";
import Timeline from "@/pages/timeline";
import Comparison from "@/pages/comparison";
import RawData from "@/pages/raw-data";
import Evaluation from "@/pages/evaluation";
import Login from "@/pages/auth/login";
import Signup from "@/pages/auth/signup";
import About from "@/pages/about";
import { DisasterContextProvider } from "@/context/disaster-context";
import { MainLayout } from "@/components/layout/main-layout";
import React from 'react';

// Protected Route component with simplified redirect logic
function ProtectedRoute({ component: Component, ...rest }: { component: React.ComponentType }) {
  const { user, isLoading } = useAuth();

  // Show loading state
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  // Redirect if not authenticated
  if (!user) {
    // One-time redirect to login
    if (typeof window !== 'undefined') {
      window.location.href = '/login';
    }
    return null;
  }

  // Render protected component
  return <Component {...rest} />;
}

function Router() {
  return (
    <Switch>
      <Route path="/login" component={Login} />
      <Route path="/signup" component={Signup} />
      <Route path="/about" component={About} />
      <Route path="/">
        <ProtectedRoute component={Dashboard} />
      </Route>
      <Route path="/dashboard">
        <ProtectedRoute component={Dashboard} />
      </Route>
      <Route path="/emotion-analysis">
        <ProtectedRoute component={EmotionAnalysis} />
      </Route>
      <Route path="/timeline">
        <ProtectedRoute component={Timeline} />
      </Route>
      <Route path="/comparison">
        <ProtectedRoute component={Comparison} />
      </Route>
      <Route path="/raw-data">
        <ProtectedRoute component={RawData} />
      </Route>
      <Route path="/evaluation">
        <ProtectedRoute component={Evaluation} />
      </Route>
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <DisasterContextProvider>
          <MainLayout>
            <Router />
          </MainLayout>
          <Toaster />
        </DisasterContextProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;