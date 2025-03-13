import { Switch, Route, Redirect } from "wouter";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import { AuthProvider, useAuth } from "@/context/auth-context";
import NotFound from "@/pages/not-found";
import Dashboard from "@/pages/dashboard";
import EmotionAnalysis from "@/pages/emotion-analysis";
import Timeline from "@/pages/timeline";
import Comparison from "@/pages/comparison";
import Login from "@/pages/auth/login";
import Signup from "@/pages/auth/signup";
import About from "@/pages/about";
import { DisasterContextProvider } from "@/context/disaster-context";
import { MainLayout } from "@/components/layout/main-layout";
import React from 'react';

// Protected Route component
function ProtectedRoute({ component: Component, ...rest }: { component: React.ComponentType }) {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return <div>Loading...</div>;
  }

  if (!user) {
    window.location.assign('/login');
    return null;
  }

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
      <Route path="/emotion-analysis">
        <ProtectedRoute component={EmotionAnalysis} />
      </Route>
      <Route path="/timeline">
        <ProtectedRoute component={Timeline} />
      </Route>
      <Route path="/comparison">
        <ProtectedRoute component={Comparison} />
      </Route>
      {/* Fallback to 404 */}
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