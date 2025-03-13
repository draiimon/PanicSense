import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from "@/components/ui/sonner";
import { MainLayout } from "@/components/layout/main-layout";
import { Router } from "@/router";
import { DisasterContextProvider } from "@/context/disaster-context";
import { UploadProvider } from "@/context/upload-context";
import { UploadIndicator } from "@/components/upload-indicator";

// Create a react-query client
const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <DisasterContextProvider>
        <UploadProvider>
          <Toaster />
          <UploadIndicator />
          <MainLayout>
            <Router />
          </MainLayout>
        </UploadProvider>
      </DisasterContextProvider>
    </QueryClientProvider>
  );
}

export default App;