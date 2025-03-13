import { Switch, Route, useLocation } from "wouter";
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
  const [, setLocation] = useLocation();

  React.useEffect(() => {
    if (!isLoading && !user) {
      window.location.href = '/login';
    }
  }, [user, isLoading, setLocation]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return <Component {...rest} />;
}

function Router() {
  const { user } = useAuth();
  const [, setLocation] = useLocation();

  // Redirect to dashboard if user is already logged in and tries to access login/signup
  React.useEffect(() => {
    if (user && (window.location.pathname === '/login' || window.location.pathname === '/signup')) {
      window.location.href = '/dashboard';
    }
  }, [user, setLocation]);

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